cmake_minimum_required(VERSION 3.21)

project(ScholarVault VERSION 0.1.0 LANGUAGES CXX)

find_package(SQLite3 REQUIRED)

option(SCHOLARVAULT_BUILD_GUI "Build the Qt 6 desktop application" ON)
option(SCHOLARVAULT_BUILD_TESTS "Build ScholarVault core tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(scholarvault_core
    src/archives.cpp
    src/arxiv.cpp
    src/domain.cpp
    src/github.cpp
    src/storage.cpp
    src/zotero.cpp
)
target_include_directories(scholarvault_core PUBLIC include)
target_link_libraries(scholarvault_core PUBLIC SQLite::SQLite3)
target_compile_options(scholarvault_core PRIVATE -Wall -Wextra -Wpedantic)

if(SCHOLARVAULT_BUILD_GUI)
    find_package(Qt6 6.2 QUIET COMPONENTS Concurrent Core Gui Network Pdf PdfWidgets Widgets Xml)
    if(Qt6_FOUND)
        find_package(X11 REQUIRED)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTOUIC ON)
        set(CMAKE_AUTORCC ON)
        qt_add_executable(scholarvault
            src/main.cpp
            src/ui/arxiv_import_task.cpp
            src/ui/assistant_panel.cpp
            src/ui/chatgpt_upload_task.cpp
            src/ui/github_clone_task.cpp
            src/ui/library_model.cpp
            src/ui/latex_preview_task.cpp
            src/ui/main_window.cpp
            src/ui/pdf_document_view.cpp
            src/ui/project_dialogs.cpp
            src/ui/reader.cpp
            src/ui/task_panel.cpp
            src/ui/zotero_catalog_loader.cpp
            src/ui/zotero_import_dialog.cpp
            src/ui/arxiv_import_task.hpp
            src/ui/assistant_panel.hpp
            src/ui/chatgpt_upload_task.hpp
            src/ui/github_clone_task.hpp
            src/ui/library_model.hpp
            src/ui/latex_preview_task.hpp
            src/ui/main_window.hpp
            src/ui/pdf_document_view.hpp
            src/ui/project_dialogs.hpp
            src/ui/reader.hpp
            src/ui/task_panel.hpp
            src/ui/zotero_catalog_loader.hpp
            src/ui/zotero_import_dialog.hpp
        )
        target_include_directories(scholarvault PRIVATE include src)
        target_link_libraries(scholarvault PRIVATE
            scholarvault_core
            Qt6::Concurrent
            Qt6::Core
            Qt6::Gui
            Qt6::Network
            Qt6::Pdf
            Qt6::PdfWidgets
            Qt6::Widgets
            Qt6::Xml
            X11::X11
        )
        set_target_properties(scholarvault PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
        )
        install(TARGETS scholarvault RUNTIME DESTINATION bin)
        install(PROGRAMS scripts/scholarvault-chatgpt-upload DESTINATION libexec)
    else()
        message(WARNING
            "Qt 6 development packages were not found. Building the tested core only. "
            "Install qt6-base-dev and qt6-pdf-dev to build the scholarvault command."
        )
    endif()
endif()

if(SCHOLARVAULT_BUILD_TESTS)
    include(CTest)
    find_path(CATCH2_INCLUDE_DIR catch2/catch.hpp REQUIRED)
    add_executable(scholarvault_core_tests tests/core_tests.cpp)
    target_include_directories(scholarvault_core_tests PRIVATE ${CATCH2_INCLUDE_DIR})
    target_link_libraries(scholarvault_core_tests PRIVATE scholarvault_core)
    add_test(NAME scholarvault_core_tests COMMAND scholarvault_core_tests)

    if(Qt6_FOUND)
        find_package(Qt6 6.2 REQUIRED COMPONENTS Test)
        qt_add_executable(scholarvault_latex_tests
            tests/latex_preview_task_test.cpp
            src/ui/latex_preview_task.cpp
            src/ui/latex_preview_task.hpp
        )
        target_include_directories(scholarvault_latex_tests PRIVATE src)
        target_link_libraries(scholarvault_latex_tests PRIVATE
            Qt6::Concurrent
            Qt6::Core
            Qt6::Test
        )
        add_test(NAME scholarvault_latex_tests COMMAND scholarvault_latex_tests)

        qt_add_executable(scholarvault_zotero_ui_tests
            tests/zotero_ui_tests.cpp
            src/ui/project_dialogs.cpp
            src/ui/project_dialogs.hpp
            src/ui/zotero_catalog_loader.cpp
            src/ui/zotero_catalog_loader.hpp
            src/ui/zotero_import_dialog.cpp
            src/ui/zotero_import_dialog.hpp
        )
        target_include_directories(scholarvault_zotero_ui_tests PRIVATE include src)
        target_link_libraries(scholarvault_zotero_ui_tests PRIVATE
            scholarvault_core
            Qt6::Concurrent
            Qt6::Core
            Qt6::Gui
            Qt6::Network
            Qt6::Test
            Qt6::Widgets
        )
        add_test(NAME scholarvault_zotero_ui_tests COMMAND scholarvault_zotero_ui_tests)
        set_tests_properties(scholarvault_zotero_ui_tests PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        qt_add_executable(scholarvault_pdf_view_tests
            tests/pdf_document_view_test.cpp
            src/ui/pdf_document_view.cpp
            src/ui/pdf_document_view.hpp
        )
        target_include_directories(scholarvault_pdf_view_tests PRIVATE src)
        target_link_libraries(scholarvault_pdf_view_tests PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Pdf
            Qt6::Test
            Qt6::Widgets
        )
        add_test(NAME scholarvault_pdf_view_tests COMMAND scholarvault_pdf_view_tests)
        set_tests_properties(scholarvault_pdf_view_tests PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen;QT_SCALE_FACTOR=2"
        )

        qt_add_executable(scholarvault_reader_actions_tests
            tests/reader_actions_test.cpp
            src/ui/pdf_document_view.cpp
            src/ui/pdf_document_view.hpp
            src/ui/reader.cpp
            src/ui/reader.hpp
        )
        target_include_directories(scholarvault_reader_actions_tests PRIVATE include src)
        target_link_libraries(scholarvault_reader_actions_tests PRIVATE
            scholarvault_core
            Qt6::Core
            Qt6::Gui
            Qt6::Pdf
            Qt6::Test
            Qt6::Widgets
        )
        add_test(NAME scholarvault_reader_actions_tests COMMAND scholarvault_reader_actions_tests)
        set_tests_properties(scholarvault_reader_actions_tests PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        qt_add_executable(scholarvault_arxiv_import_tests
            tests/arxiv_import_task_test.cpp
            src/ui/arxiv_import_task.cpp
            src/ui/arxiv_import_task.hpp
        )
        target_include_directories(scholarvault_arxiv_import_tests PRIVATE include src)
        target_link_libraries(scholarvault_arxiv_import_tests PRIVATE
            scholarvault_core
            Qt6::Core
            Qt6::Network
            Qt6::Test
            Qt6::Xml
        )
        add_test(NAME scholarvault_arxiv_import_tests
                 COMMAND scholarvault_arxiv_import_tests)

        qt_add_executable(scholarvault_assistant_panel_tests
            tests/assistant_panel_test.cpp
            src/ui/assistant_panel.cpp
            src/ui/assistant_panel.hpp
            src/ui/chatgpt_upload_task.cpp
            src/ui/chatgpt_upload_task.hpp
        )
        target_include_directories(scholarvault_assistant_panel_tests PRIVATE include src)
        target_link_libraries(scholarvault_assistant_panel_tests PRIVATE
            Qt6::Core
            Qt6::Gui
            Qt6::Test
            Qt6::Widgets
            X11::X11
        )
        add_test(NAME scholarvault_assistant_panel_tests
                 COMMAND scholarvault_assistant_panel_tests)
        set_tests_properties(scholarvault_assistant_panel_tests PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen;QT_SCALE_FACTOR=1.25"
        )

        qt_add_executable(scholarvault_main_window_selection_tests
            tests/main_window_selection_test.cpp
            src/ui/arxiv_import_task.cpp
            src/ui/arxiv_import_task.hpp
            src/ui/assistant_panel.cpp
            src/ui/assistant_panel.hpp
            src/ui/github_clone_task.cpp
            src/ui/github_clone_task.hpp
            src/ui/library_model.cpp
            src/ui/library_model.hpp
            src/ui/latex_preview_task.cpp
            src/ui/latex_preview_task.hpp
            src/ui/main_window.cpp
            src/ui/main_window.hpp
            src/ui/pdf_document_view.cpp
            src/ui/pdf_document_view.hpp
            src/ui/project_dialogs.cpp
            src/ui/project_dialogs.hpp
            src/ui/reader.cpp
            src/ui/reader.hpp
            src/ui/task_panel.cpp
            src/ui/task_panel.hpp
            src/ui/zotero_catalog_loader.cpp
            src/ui/zotero_catalog_loader.hpp
            src/ui/zotero_import_dialog.cpp
            src/ui/zotero_import_dialog.hpp
        )
        target_include_directories(scholarvault_main_window_selection_tests PRIVATE
            include src)
        target_link_libraries(scholarvault_main_window_selection_tests PRIVATE
            scholarvault_core
            Qt6::Concurrent
            Qt6::Core
            Qt6::Gui
            Qt6::Network
            Qt6::Pdf
            Qt6::Test
            Qt6::Widgets
            Qt6::Xml
            X11::X11
        )
        add_test(NAME scholarvault_main_window_selection_tests
                 COMMAND scholarvault_main_window_selection_tests)
        set_tests_properties(scholarvault_main_window_selection_tests PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )
    endif()
endif()
