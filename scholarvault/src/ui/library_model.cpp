#include "ui/library_model.hpp"

#include <QBrush>
#include <QColor>
#include <QFileIconProvider>

#include <algorithm>
#include <functional>

namespace scholarvault::ui {

LibraryModel::LibraryModel(QObject* parent)
    : QAbstractItemModel(parent), root_(std::make_unique<Node>()) {
    root_->childrenLoaded = true;
}

int LibraryModel::Node::row() const {
    if (parent == nullptr) return 0;
    const auto found = std::find_if(parent->children.begin(), parent->children.end(),
                                    [this](const auto& item) { return item.get() == this; });
    return found == parent->children.end()
               ? 0
               : static_cast<int>(std::distance(parent->children.begin(), found));
}

void LibraryModel::setStorage(std::shared_ptr<VaultStorage> storage) {
    storage_ = std::move(storage);
    reload();
}

void LibraryModel::reload() {
    beginResetModel();
    root_ = std::make_unique<Node>();
    root_->childrenLoaded = true;
    if (storage_) {
        for (const auto& topic : storage_->discoverTopics()) {
            auto node = std::make_unique<Node>();
            node->type = NodeType::Topic;
            node->parent = root_.get();
            node->topic = topic;
            root_->children.push_back(std::move(node));
        }
    }
    endResetModel();
}

LibraryModel::Node* LibraryModel::nodeForIndex(const QModelIndex& index) const {
    return index.isValid() ? static_cast<Node*>(index.internalPointer()) : root_.get();
}

QModelIndex LibraryModel::index(int row, int column, const QModelIndex& parentIndex) const {
    if (row < 0 || column != 0) return {};
    Node* parent = nodeForIndex(parentIndex);
    if (parent == nullptr || row >= static_cast<int>(parent->children.size())) return {};
    return createIndex(row, column, parent->children[static_cast<std::size_t>(row)].get());
}

QModelIndex LibraryModel::parent(const QModelIndex& child) const {
    if (!child.isValid()) return {};
    Node* node = nodeForIndex(child);
    if (node == nullptr || node->parent == nullptr || node->parent == root_.get()) return {};
    return createIndex(node->parent->row(), 0, node->parent);
}

int LibraryModel::rowCount(const QModelIndex& parentIndex) const {
    if (parentIndex.column() > 0) return 0;
    const Node* parent = nodeForIndex(parentIndex);
    return parent == nullptr ? 0 : static_cast<int>(parent->children.size());
}

int LibraryModel::columnCount(const QModelIndex&) const { return 1; }

QVariant LibraryModel::data(const QModelIndex& modelIndex, int role) const {
    if (!modelIndex.isValid()) return {};
    const Node* node = nodeForIndex(modelIndex);
    if (node == nullptr) return {};
    if (role == Qt::DisplayRole) {
        if (node->topic) return QString::fromStdString(node->topic->name);
        if (node->project) return QString::fromStdString(node->project->title);
    }
    if (role == NodeTypeRole) return static_cast<int>(node->type);
    if (role == PathRole) {
        if (node->topic) return QString::fromStdString(node->topic->path.string());
        if (node->project) return QString::fromStdString(node->project->path.string());
    }
    if (role == IdentifierRole) {
        if (node->topic) return QString::fromStdString(node->topic->id);
        if (node->project) return QString::fromStdString(node->project->id);
    }
    if (role == Qt::ForegroundRole && node->project &&
        !std::filesystem::exists(node->project->path / node->project->pdfRelativePath)) {
        return QBrush(QColor("#b45309"));
    }
    return {};
}

bool LibraryModel::hasChildren(const QModelIndex& parentIndex) const {
    const Node* node = nodeForIndex(parentIndex);
    if (node == nullptr) return false;
    return node->type == NodeType::Root || node->type == NodeType::Topic;
}

bool LibraryModel::canFetchMore(const QModelIndex& parentIndex) const {
    const Node* node = nodeForIndex(parentIndex);
    return storage_ && node && node->type == NodeType::Topic && !node->childrenLoaded;
}

void LibraryModel::fetchMore(const QModelIndex& parentIndex) {
    Node* node = nodeForIndex(parentIndex);
    if (!storage_ || node == nullptr || node->type != NodeType::Topic ||
        node->childrenLoaded || !node->topic) {
        return;
    }
    const auto topics = storage_->discoverChildTopics(*node->topic);
    const auto projects = storage_->discoverProjects(*node->topic);
    const std::size_t childCount = topics.size() + projects.size();
    if (childCount > 0) {
        beginInsertRows(parentIndex, 0, static_cast<int>(childCount) - 1);
        for (const auto& topic : topics) {
            auto child = std::make_unique<Node>();
            child->type = NodeType::Topic;
            child->parent = node;
            child->topic = topic;
            node->children.push_back(std::move(child));
        }
        for (const auto& project : projects) {
            auto child = std::make_unique<Node>();
            child->type = NodeType::Project;
            child->parent = node;
            child->childrenLoaded = true;
            child->project = project;
            node->children.push_back(std::move(child));
        }
        endInsertRows();
    }
    node->childrenLoaded = true;
}

std::optional<Topic> LibraryModel::topicForIndex(const QModelIndex& modelIndex) const {
    const Node* node = nodeForIndex(modelIndex);
    if (node == nullptr) return std::nullopt;
    if (node->topic) return node->topic;
    if (node->parent && node->parent->topic) return node->parent->topic;
    return std::nullopt;
}

std::optional<Project> LibraryModel::projectForIndex(const QModelIndex& modelIndex) const {
    const Node* node = nodeForIndex(modelIndex);
    return node == nullptr ? std::nullopt : node->project;
}

QModelIndex LibraryModel::indexForIdentifier(const QString& identifier) {
    if (identifier.isEmpty()) return {};
    std::function<QModelIndex(const QModelIndex&)> find =
        [this, &identifier, &find](const QModelIndex& parentIndex) -> QModelIndex {
            if (canFetchMore(parentIndex)) fetchMore(parentIndex);
            for (int row = 0; row < rowCount(parentIndex); ++row) {
                const QModelIndex child = index(row, 0, parentIndex);
                if (child.data(IdentifierRole).toString() == identifier) return child;
                if (child.data(NodeTypeRole).toInt() ==
                    static_cast<int>(NodeType::Topic)) {
                    const QModelIndex nested = find(child);
                    if (nested.isValid()) return nested;
                }
            }
            return {};
        };
    return find({});
}

} // namespace scholarvault::ui
