"""widget for managing application wide settings"""

import importlib.resources
import sys

from PyQt5 import QtCore, QtWidgets

from .settings import SettingsItem, settings

_MODULE = sys.modules[__package__]
_FILES = importlib.resources.files(_MODULE)
_INFO_LABEL_PATH = _FILES.joinpath("infolabel.html")

NAME_INDEX = 0
VALUE_INDEX = 1
RESET_INDEX = 2
DEFAULT_INDEX = 3
COMMENT_INDEX = 4


class SettingsWidgetUI(QtWidgets.QWidget):
    """widget for managing application wide settings

    parent: parent of this qt object
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.gridlayout = QtWidgets.QGridLayout(self)

        self.treeWidget = QtWidgets.QTreeWidget(self)
        self.treeWidget.headerItem().setText(NAME_INDEX, "Name")
        self.treeWidget.header().resizeSection(NAME_INDEX, 250)
        self.treeWidget.headerItem().setText(VALUE_INDEX, "Value")
        self.treeWidget.headerItem().setText(RESET_INDEX, "Reset")
        self.treeWidget.header().resizeSection(RESET_INDEX, 20)
        self.treeWidget.headerItem().setText(DEFAULT_INDEX, "Default")
        self.treeWidget.headerItem().setText(COMMENT_INDEX, "Comment")

        self.editorDelegate = TreeWidgetEditorDelegate(self.treeWidget)
        self.treeWidget.setItemDelegate(self.editorDelegate)
        self.treeWidget.itemChanged.connect(item_changed)
        self.gridlayout.addWidget(self.treeWidget, 0, 0, 1, 3)

        self.label = QtWidgets.QLabel(self)
        with _INFO_LABEL_PATH.open() as fp:
            self.label.setText(fp.read())

        self.gridlayout.addWidget(self.label, 1, 0, 3, 1)

        self.collapseButton = QtWidgets.QPushButton("collapse all", self)
        self.collapseButton.clicked.connect(self.treeWidget.collapseAll)
        self.gridlayout.addWidget(self.collapseButton, 1, 1, 1, 1)

        self.expandButton = QtWidgets.QPushButton("expand all", self)
        self.expandButton.clicked.connect(self.treeWidget.expandAll)
        self.gridlayout.addWidget(self.expandButton, 1, 2, 1, 1)

        self.resetButton = QtWidgets.QPushButton("reset all to default", self)
        self.gridlayout.addWidget(self.resetButton, 2, 1, 1, 1)

        self.updateButton = QtWidgets.QPushButton("cancel", self)
        self.updateButton.clicked.connect(self.update)
        self.gridlayout.addWidget(self.updateButton, 2, 2, 1, 1)

        self.saveButton = QtWidgets.QPushButton("save", self)
        self.gridlayout.addWidget(self.saveButton, 3, 2, 1, 1)

    def update(self):
        self.treeWidget.itemChanged.disconnect(item_changed)
        self.treeWidget.clear()
        self.read_item(settings(), self.treeWidget)
        self.treeWidget.expandAll()
        self.treeWidget.itemChanged.connect(item_changed)
        for index in NAME_INDEX, VALUE_INDEX, DEFAULT_INDEX, COMMENT_INDEX:
            self.treeWidget.resizeColumnToContents(index)

    def read_item(self, settings_item, parent):
        for key, value in settings_item.items():
            tree_item = QtWidgets.QTreeWidgetItem(parent)
            tree_item.setText(NAME_INDEX, key)
            if type(value) is SettingsItem:
                self.read_item(value, tree_item)
            else:
                tree_item.setText(VALUE_INDEX, str(value))
                default = settings_item.defaults[key]
                button = QtWidgets.QPushButton("R")
                self.treeWidget.setItemWidget(tree_item, RESET_INDEX, button)
                tree_item.setText(DEFAULT_INDEX, str(default))
                tree_item.setText(COMMENT_INDEX, "")
                flags = tree_item.flags()
                tree_item.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEditable)


def item_changed(item, column):
    if column == VALUE_INDEX:
        font = item.font(VALUE_INDEX)
        font.setBold(True)
        item.setFont(VALUE_INDEX, font)


class TreeWidgetEditorDelegate(QtWidgets.QItemDelegate):
    def __init__(self, parent):
        self.treeWidget = parent
        super().__init__(parent)

    def createEditor(self, parent, option, index):  # override
        if index.column() != VALUE_INDEX:
            item = self.treeWidget.itemFromIndex(index)
            self.treeWidget.editItem(item, VALUE_INDEX)
            return None

        return super().createEditor(parent, option, index)
