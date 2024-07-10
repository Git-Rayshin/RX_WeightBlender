from functools import partial

from maya import cmds, OpenMaya as om

from .operations import _collect_data, _solve, _check_selection, get_minimum_shared_infls_by_vertex, \
    _match_minimum_infls, _solve_two_source, _blend_half
# --- UTILS ---
from .utils import maya_main_window, DPI_SCALE, PY2, QtCore, QtWidgets, QtGui, showDialog

from .utils.helpers import preserveSelection, undoable

label_styles = """
QWidget{
    font-size: {}px;
} 
"""


class _SliderBtn(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(_SliderBtn, self).__init__(parent)

        self.pixmap_background = QtGui.QPixmap("mgear_folder.svg")
        self.resize(self.pixmap_background.size())
        self.text = ""

    def set_text(self, text):
        self.text = text
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter()
        p.begin(self)

        p.drawPixmap(
            self.rect(),
            self.pixmap_background.scaled(self.size())
        )


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Plain)


class _LineEdit(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super(_LineEdit, self).__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setup_validator()

    def _on_finished(self):
        self.hide()
        self.parent().left_clicked.emit()
        self.parent().set_value(float(self.text()))
        self.parent().left_released.emit()

    def setup_validator(self):
        validator = QtGui.QDoubleValidator(self)
        validator.setRange(
            self.parent().min_value(),
            self.parent().max_value()
        )
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator.setDecimals(2)
        self.setValidator(validator)

    def showEvent(self, event):
        self.setText("{:.2f}".format(self.parent().value()))
        self.selectAll()
        self.setFocus()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter:
            self._on_finished()
        else:
            super(_LineEdit, self).keyPressEvent(event)


class SliderAbsolutePosition(QtWidgets.QWidget):
    value_changed = QtCore.Signal(float)
    left_clicked = QtCore.Signal()
    left_released = QtCore.Signal()

    def __init__(self, parent=None):
        super(SliderAbsolutePosition, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self._value = 0.0
        self._default_value = 0.0
        self._min_value = 0.0
        self._max_value = 100.0

        self._le_value = _LineEdit(self)
        self._le_value.hide()

        self._is_value_changing = False
        self.fake_focus = True

        self._last_click = None

    def default_value(self):
        return self._default_value

    def set_default_value(self, value):
        self._default_value = value

    def value(self):
        return self._value

    def _set_value(self, value):
        self._value = max(min(value, self._max_value), self._min_value)
        self.value_changed.emit(self._value)

    def set_value(self, value):
        # print("setting value:%s" % value)
        self._set_value(float(value))
        self.update()

    def min_value(self):
        return self._min_value

    def set_min_value(self, value):
        self._min_value = value
        self._le_value.setup_validator()

    def max_value(self):
        return self._max_value

    def set_max_value(self, value):
        self._max_value = value
        self._le_value.setup_validator()

    def resizeEvent(self, event):
        self._le_value.resize(int(self.width() * 0.5), int(self.height() * 0.8))
        self._le_value.move(int(self.width() * (1 - 0.5) / 2), int(self.height() * (1 - 0.8) / 2 + 1))
        font = QtGui.QFont()
        font.setPixelSize(int(self.height() / 2))
        self._le_value.setFont(font)

    def mousePressEvent(self, event):
        range_scale = (self.max_value() - self.min_value()) / 100.0
        offset = self.min_value()
        if event.button() == QtCore.Qt.LeftButton:
            self._last_click = "CLICK"
            self._is_value_changing = True
            rel_pos = self.mapFromGlobal(QtGui.QCursor.pos()).x()
            val = min(max((rel_pos / float(self.width()) * 100) * range_scale + offset, self.min_value()),
                      self._max_value)
            if event.modifiers() == QtCore.Qt.ControlModifier:
                w_step = 5.0
                val = round(val / w_step) * w_step
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                w_step = 10.0
                val = round(val / w_step) * w_step
            if event.modifiers() == QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier:
                w_step = 20.0
                val = round(val / w_step) * w_step
            if self.fake_focus:
                self.set_value(val)
            self.left_clicked.emit()
        if event.button() == QtCore.Qt.RightButton:
            self._le_value.show()
            self.fake_focus = False
        if event.button() == QtCore.Qt.MiddleButton:
            self.set_value(self.default_value())

    def mouseMoveEvent(self, event):
        range_scale = (self.max_value() - self.min_value()) / 100.0
        offset = self.min_value()
        if self._is_value_changing:
            self.setCursor(QtCore.Qt.SizeHorCursor)
            rel_pos = self.mapFromGlobal(QtGui.QCursor.pos()).x()
            val = min(max(rel_pos / float(self.width()) * 100.0 * range_scale + offset, self.min_value()),
                      self._max_value)
            if event.modifiers() == QtCore.Qt.ControlModifier:
                w_step = 5.0
                val = round(val / w_step) * w_step
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                w_step = 10.0
                val = round(val / w_step) * w_step
            if event.modifiers() == QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier:
                w_step = 20.0
                val = round(val / w_step) * w_step
            self.set_value(val)

    def mouseReleaseEvent(self, event):
        range_scale = (self.max_value() - self.min_value()) / 100.0
        # range_scale = 1
        offset = self.min_value()
        if event.button() == QtCore.Qt.LeftButton:
            self.setCursor(QtCore.Qt.ArrowCursor)
            self._is_value_changing = False
            rel_pos = float(self.mapFromGlobal(QtGui.QCursor.pos()).x())
            val = min(max(rel_pos / float(self.width()) * 100.0 * range_scale + offset, self.min_value()),
                      self._max_value)
            if event.modifiers() == QtCore.Qt.ControlModifier:
                w_step = 5.0
                val = round(val / w_step) * w_step
            if event.modifiers() == QtCore.Qt.ShiftModifier:
                w_step = 10.0
                val = round(val / w_step) * w_step
            if event.modifiers() == QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier:
                w_step = 20.0
                val = round(val / w_step) * w_step
            if self.fake_focus:
                self.set_value(val)
            elif not self.fake_focus and not self._last_click == "DOUBLE":
                self.fake_focus = True
            self.left_released.emit()

    def mouseDoubleClickEvent(self, event):
        pass
        # if event.button() == QtCore.Qt.LeftButton:
        #     self._last_click = "DOUBLE"
        #     self._le_value.show()
        #     self.fake_focus = False

    def paintEvent(self, event):
        range_ = self.max_value() - self.min_value()
        range_scale = range_ / 100.0
        offset = self.width() / 100.0 * self.min_value()
        p = QtGui.QPainter()

        pix = QtGui.QPixmap(self.size())
        pix.fill(QtCore.Qt.transparent)
        p.begin(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # gray bg
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(200, 200, 200)))
        # p.setBrush(QtGui.QBrush(QtGui.QColor(112,112,112)))
        p.drawRect(self.rect())

        # progress bar
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(93, 93, 160)))

        # p.setBrush(QtGui.QBrush(QtGui.QColor(108, 108, 162)))
        start_point = (self.default_value() - self.min_value()) / (self.max_value() - self.min_value()) * self.width()
        shift = int(self.width() / range_ * self._value)

        p.drawRect(QtCore.QRect(
            start_point,
            0,
            shift,
            self.height()
        ))

        p.setBrush(QtGui.QBrush(QtGui.QColor(90, 90, 90)))
        p.drawRoundedRect(QtCore.QRect(
            start_point + shift - self.height() - 4,
            2,
            self.height() * 2 + 8,
            self.height() - 4
        ), (self.height() - 4) / 2, (self.height() - 4) / 2)

        p.setBrush(QtGui.QBrush(QtGui.QColor(150, 150, 150)))
        p.drawRoundedRect(QtCore.QRect(
            start_point + shift - self.height() - 3,
            3,
            self.height() * 2 + 6,
            self.height() - 6
        ), (self.height() - 6) / 2, (self.height() - 6) / 2)

        p.setBrush(QtGui.QBrush(QtGui.QColor(130, 130, 130)))
        p.drawRoundedRect(QtCore.QRect(
            start_point + shift - self.height() - 2,
            4,
            self.height() * 2 + 4,
            self.height() - 8
        ), (self.height() - 8) / 2, (self.height() - 8) / 2)

        # gradient texture
        g = QtGui.QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, QtGui.QColor(255, 255, 255, 100))
        g.setColorAt(0.2, QtGui.QColor(255, 255, 255, 80))
        g.setColorAt(0.45, QtGui.QColor(255, 255, 255, 40))
        g.setColorAt(0.451, QtGui.QColor(0, 0, 0, 0))

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(g)
        p.drawRect(self.rect())

        # slider values
        font = QtGui.QFont()
        font.setPixelSize(int(self.height() / 2))
        p.setFont(font)

        # p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20)))
        # p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
        p.setBrush(QtCore.Qt.NoBrush)
        textRefRect = QtCore.QRect(
            start_point + shift - self.height(),
            0,
            self.height() * 2,
            self.height()
        )
        p.drawText(
            textRefRect, QtCore.Qt.AlignCenter, "{:.2f}%".format(self._value)
        )
        # p.drawText(
        #     self.rect(), QtCore.Qt.AlignCenter, "{:.2f}%".format(self._value)
        # )

        # rounded frame
        r = self.height() / 2
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(150, 150, 150))
        pen.setWidth(5)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(self.rect(), r, r)

        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(200, 200, 200))
        pen.setWidth(3)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(self.rect(), r, r)

        # clear
        p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtCore.Qt.red)

        # # Top Left
        path = QtGui.QPainterPath()
        path.moveTo(0, 0)
        path.lineTo(r, 0)
        path.arcTo(QtCore.QRectF(0, 0, r * 2, r * 2), 90, 90)
        path.closeSubpath()
        p.drawPath(path)
        w = self.width()
        h = self.height()
        # # Bottom Left
        path = QtGui.QPainterPath()
        path.moveTo(0, h)
        path.lineTo(0, h - r)
        path.arcTo(QtCore.QRectF(0, h - r * 2, r * 2, r * 2), 180, 90)
        path.closeSubpath()
        p.drawPath(path)
        # # Bottom Right
        path = QtGui.QPainterPath()
        path.moveTo(w, h)
        path.lineTo(w - r, h)
        path.arcTo(QtCore.QRectF(w - r * 2, h - r * 2, r * 2, r * 2), 270, 90)
        path.closeSubpath()
        p.drawPath(path)
        # # Top Right
        path = QtGui.QPainterPath()
        path.moveTo(w, 0)
        path.lineTo(w, r)
        path.arcTo(QtCore.QRectF(w - r * 2, 0, r * 2, r * 2), 0, 90)
        path.closeSubpath()
        p.drawPath(path)

        p.end()
        p.begin(self)
        p.drawPixmap(self.rect(), pix)
        p.end()


class SliderGroup(QtWidgets.QHBoxLayout):

    def __init__(self):
        QtWidgets.QHBoxLayout.__init__(self)
        self.slider = SliderAbsolutePosition()
        self.addWidget(self.slider)
        self.slider.setFixedHeight(20 * DPI_SCALE)


class WeightBlendWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(WeightBlendWidget, self).__init__(parent)

        self.data = None
        self.create_widgets()
        self.create_layout()
        self.create_connections()

    def create_widgets(self):
        self.match_minimum_infls_btn = QtWidgets.QPushButton("Match Minimum Infls")

        self.top_line = QHLine()

        self.first_slider_lb = QtWidgets.QLabel("Blend Weight")
        self.first_slider = SliderGroup()

        self.mid_line = QHLine()

        self.second_slider_lb = QtWidgets.QLabel("Two Source Blend Weight")
        self.half_blend_btn = QtWidgets.QPushButton("Blend 50%")
        self.second_slider = SliderGroup()
        self.second_slider.slider.set_min_value(-100)

    def create_layout(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.match_minimum_infls_btn)

        main_layout.addWidget(self.top_line)

        main_layout.addWidget(self.first_slider_lb)
        main_layout.addLayout(self.first_slider)

        main_layout.addWidget(self.mid_line)

        main_layout.addWidget(self.second_slider_lb)
        main_layout.addWidget(self.half_blend_btn)
        main_layout.addLayout(self.second_slider)

    def create_connections(self):
        # top btn
        self.match_minimum_infls_btn.clicked.connect(self.match_minimum_infls)

        # first slider
        self.first_slider.slider.left_clicked.connect(self.collect_data)
        self.first_slider.slider.value_changed.connect(self.first_solve)
        self.first_slider.slider.left_released.connect(self.data_clear_first)

        # second slider
        self.half_blend_btn.clicked.connect(self.half_blend)
        self.second_slider.slider.left_clicked.connect(partial(self.collect_data, targetNum=2))
        self.second_slider.slider.value_changed.connect(self.second_solve)
        self.second_slider.slider.left_released.connect(self.data_clear_second)

    def collect_data(self, pairs=False, targetNum=1):
        if not cmds.ls(sl=1, io=1) or len(cmds.ls(sl=1, fl=1, io=1)) < 2:
            return om.MGlobal.displayWarning("must select at least 2 verts")
        # print("collecting")
        self.data = _collect_data(pairs, targetNum)

    @staticmethod
    @undoable
    @preserveSelection
    def match_minimum_infls():
        try:
            all_dict = _check_selection()[2]
        except TypeError:
            return
        dagArray, shared_infl_names = get_minimum_shared_infls_by_vertex(all_dict)
        _match_minimum_infls(dagArray, shared_infl_names)

    def first_solve(self):
        if not self.data:
            return
        _solve(self.data, self.first_slider.slider.value() / 100.0)

    def second_solve(self):
        if not self.data:
            return
        _solve_two_source(self.data, self.second_slider.slider.value() / 100.0)

    def half_blend(self):
        self.collect_data(targetNum=2)
        _blend_half(self.data)
        self.data = None

    def data_clear_first(self):
        if self.data:
            self.data = None
            # print("data cleared")
        self.first_slider.slider.set_value(0.0)

    def data_clear_second(self):
        if self.data:
            self.data = None
            # print("data cleared")
        self.second_slider.slider.set_value(0.0)


class WeightBlendDialog(QtWidgets.QDialog):

    def __init__(self, parent=maya_main_window()):
        super(WeightBlendDialog, self).__init__(parent)
        self.setWindowTitle("Weight Blender")
        # remove the question mark of a QDialog object
        if PY2:
            self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(418 * DPI_SCALE, 177 * DPI_SCALE)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(WeightBlendWidget())


def show(dock=False):
    showDialog(WeightBlendDialog, dockable=dock)


if __name__ == '__main__':
    show(dock=True)
