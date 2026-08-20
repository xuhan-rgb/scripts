#pragma once

#include "scholarvault/storage.hpp"

#include <QAbstractItemModel>

#include <memory>
#include <vector>

namespace scholarvault::ui {

class LibraryModel final : public QAbstractItemModel {
    Q_OBJECT

public:
    enum class NodeType { Root, Topic, Project };
    enum Role { NodeTypeRole = Qt::UserRole + 1, PathRole, IdentifierRole };

    explicit LibraryModel(QObject* parent = nullptr);

    void setStorage(std::shared_ptr<VaultStorage> storage);
    void reload();

    [[nodiscard]] QModelIndex index(int row, int column,
                                    const QModelIndex& parent = {}) const override;
    [[nodiscard]] QModelIndex parent(const QModelIndex& child) const override;
    [[nodiscard]] int rowCount(const QModelIndex& parent = {}) const override;
    [[nodiscard]] int columnCount(const QModelIndex& parent = {}) const override;
    [[nodiscard]] QVariant data(const QModelIndex& index,
                                int role = Qt::DisplayRole) const override;
    [[nodiscard]] bool hasChildren(const QModelIndex& parent = {}) const override;
    [[nodiscard]] bool canFetchMore(const QModelIndex& parent) const override;
    void fetchMore(const QModelIndex& parent) override;

    [[nodiscard]] std::optional<Topic> topicForIndex(const QModelIndex& index) const;
    [[nodiscard]] std::optional<Project> projectForIndex(const QModelIndex& index) const;
    [[nodiscard]] QModelIndex indexForIdentifier(const QString& identifier);

private:
    struct Node {
        NodeType type{NodeType::Root};
        Node* parent{nullptr};
        bool childrenLoaded{false};
        std::optional<Topic> topic;
        std::optional<Project> project;
        std::vector<std::unique_ptr<Node>> children;

        [[nodiscard]] int row() const;
    };

    [[nodiscard]] Node* nodeForIndex(const QModelIndex& index) const;

    std::shared_ptr<VaultStorage> storage_;
    std::unique_ptr<Node> root_;
};

} // namespace scholarvault::ui
