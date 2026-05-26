import sys
import os
os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"

import socket
import threading
from datetime import datetime
import html
import traceback
import hashlib

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QMessageBox,
    QSizePolicy, QSplitter, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem
)

from detection import init_models, detect_and_recognize_image, save_result_image


# =========================
# 工具函数
# =========================
SOCKET_IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def normalize_socket_path(raw_path: str) -> str:
    """
    把 socket 收到的路径尽量纠正成正常 Windows 路径
    """
    if raw_path is None:
        return ""

    s = raw_path.strip().strip('"').strip("'")

    restore_map = {
        "\x0c": r"\f",
        "\t": r"\t",
        "\n": r"\n",
        "\r": r"\r",
        "\b": r"\b",
        "\a": r"\a",
        "\v": r"\v",
    }
    for bad, rep in restore_map.items():
        s = s.replace(bad, rep)

    s = s.replace("/", "\\")

    if s.startswith("\\\\"):
        head = "\\\\"
        body = s[2:]
        while "\\\\" in body:
            body = body.replace("\\\\", "\\")
        s = head + body
    else:
        while "\\\\" in s:
            s = s.replace("\\\\", "\\")

    return os.path.normpath(s)


def validate_socket_image_path(raw_path: str):
    """
    返回:
    (True, fixed_path)  -> 校验通过
    (False, err_code)   -> 校验失败
    """
    fixed_path = normalize_socket_path(raw_path)

    if not fixed_path:
        return False, "ERR_EMPTY_PATH"

    if not os.path.exists(fixed_path):
        return False, "ERR_PATH_NOT_FOUND"

    if not os.path.isfile(fixed_path):
        return False, "ERR_NOT_A_FILE"

    ext = os.path.splitext(fixed_path)[1].lower()
    if ext not in SOCKET_IMG_EXTS:
        return False, "ERR_UNSUPPORTED_FORMAT"

    return True, fixed_path


def socket_msg_to_cn(code_or_msg: str) -> str:
    """
    把内部错误码转成中文说明
    """
    mapping = {
        "ERR_MODEL_NOT_READY": "模型尚未初始化完成",
        "ERR_EMPTY_PATH": "路径为空",
        "ERR_PATH_NOT_FOUND": "图片路径不存在",
        "ERR_NOT_A_FILE": "该路径不是文件",
        "ERR_UNSUPPORTED_FORMAT": "图片格式不支持，仅支持 bmp/png/jpg/jpeg/tif/tiff",
        "ERR_IMAGE_OPEN_FAILED": "图片打开失败",
        "ERR_NO_DETECTION": "未检测到镭刻码",
        "ERR_EMPTY_RESULT": "识别结果为空",
        "ERR_INTERNAL": "程序内部错误",
        "MODEL_NOT_READY": "模型尚未初始化完成",
        "None": "未识别到结果",
    }
    return mapping.get(code_or_msg, code_or_msg)


def is_socket_error_message(msg: str) -> bool:
    err_msgs = {
        "模型尚未初始化完成",
        "路径为空",
        "图片路径不存在",
        "该路径不是文件",
        "图片格式不支持，仅支持 bmp/png/jpg/jpeg/tif/tiff",
        "图片打开失败",
        "未检测到镭刻码",
        "识别结果为空",
        "程序内部错误",
        "未识别到结果",
    }
    return msg in err_msgs


DEBUG_LOG = False

def dbg_log(tag, msg):
    if not DEBUG_LOG:
        return

    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    pid = os.getpid()
    tid = threading.get_ident()
    print(f"[{now}] [PID:{pid}] [TID:{tid}] [{tag}] {msg}", flush=True)


# =========================
# stdout/stderr 重定向到界面日志
# =========================
class EmittingStream(QObject):
    text_written = Signal(str)

    def write(self, text):
        if text:
            self.text_written.emit(str(text))

    def flush(self):
        pass


# =========================
# 批量识别线程
# =========================
class BatchWorker(QThread):
    result_signal = Signal(str, dict)   # image_path, result_dict
    status_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, image_paths, output_dir):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self._paused = False
        self._stopped = False
        self.summary_lines = []

    def run(self):
        total = len(self.image_paths)
        self.summary_lines.clear()

        for i, img_path in enumerate(self.image_paths, 1):
            if self._stopped:
                break

            while self._paused and not self._stopped:
                self.msleep(200)

            if self._stopped:
                break

            try:
                self.status_signal.emit(f"批量处理中：{i}/{total}  {os.path.basename(img_path)}")

                result = detect_and_recognize_image(img_path)
                self.result_signal.emit(img_path, result)

                if result["has_result"]:
                    code_text = result["best_text"]
                else:
                    code_text = "NO_DET"

                self.summary_lines.append(f"{os.path.basename(img_path)}\t{code_text}")

                if self.output_dir:
                    save_name = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
                    save_path = os.path.join(self.output_dir, save_name)
                    save_result_image(img_path, save_path, result)

            except Exception as e:
                self.summary_lines.append(f"{os.path.basename(img_path)}\tERROR\t{str(e)}")
                self.status_signal.emit(f"批量处理失败：{os.path.basename(img_path)}")

        if self.output_dir and self.summary_lines:
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                txt_path = os.path.join(self.output_dir, f"batch_result_{stamp}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("image_name\tresult\n")
                    for line in self.summary_lines:
                        f.write(line + "\n")
                self.status_signal.emit(f"批量结果已保存：{txt_path}")
            except Exception as e:
                self.status_signal.emit(f"保存汇总失败：{e}")

        self.status_signal.emit("批量识别完成")
        self.finished_signal.emit()

    def toggle_pause(self):
        self._paused = not self._paused

    def stop(self):
        self._stopped = True
        self._paused = False


# =========================
# Socket Server Thread
# =========================
class SocketServerThread(threading.Thread):
    def __init__(self, host, port, callback):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.callback = callback
        self.server_socket = None
        self.is_running = False
        dbg_log("SocketServerThread.__init__", f"host={host}, port={port}, daemon={self.daemon}")

    def run(self):
        dbg_log("SocketServerThread.run", "thread started")
        try:
            self.is_running = True
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            dbg_log("SocketServerThread.run", f"before bind {self.host}:{self.port}")
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"SERVER_READY {self.host}:{self.port}")
            dbg_log("SocketServerThread.run", "listen success")
        except Exception as e:
            print(f"SERVER_BIND_ERROR: {e}")
            dbg_log("SocketServerThread.run", f"bind/listen failed: {e}")
            dbg_log("SocketServerThread.run", traceback.format_exc())
            return

        while self.is_running:
            client_socket = None
            try:
                dbg_log("SocketServerThread.run", "waiting accept...")
                client_socket, client_address = self.server_socket.accept()
                print(f"CLIENT_CONNECTED {client_address}")
                dbg_log("SocketServerThread.run", f"accepted client={client_address}")

                # 一个连接下，持续接收多次
                while self.is_running:
                    raw = client_socket.recv(4096)

                    # 客户端主动断开
                    if not raw:
                        print("CLIENT_DISCONNECTED")
                        dbg_log("SocketServerThread.run", "client disconnected")
                        break

                    try:
                        text = raw.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        text = raw.decode("gbk", errors="ignore").strip()

                    print(f"RX_RAW_PATH: {repr(text)}")

                    fixed_path = normalize_socket_path(text)
                    print(f"RX_FIXED_PATH: {repr(fixed_path)}")

                    result = self.callback(fixed_path)
                    dbg_log("SocketServerThread.run", f"after callback, result={result}")

                    if not result:
                        result = "未识别到结果"

                    # 返回结果，但不关闭连接
                    client_socket.sendall(result.encode("gbk", errors="ignore"))
                    print(f"SOCKET_REPLY: {result}")
                    dbg_log("SocketServerThread.run", "sendall success")

            except Exception as e:
                print(f"SERVER_RUNTIME_ERROR: {e}")
                dbg_log("SocketServerThread.run", f"runtime error: {e}")
                dbg_log("SocketServerThread.run", traceback.format_exc())
            finally:
                if client_socket:
                    try:
                        client_socket.close()
                        dbg_log("SocketServerThread.run", "client socket closed")
                    except Exception:
                        pass

    def stop(self):
        dbg_log("SocketServerThread.stop", "called")
        self.is_running = False
        if self.server_socket:
            try:
                self.server_socket.close()
                dbg_log("SocketServerThread.stop", "server socket closed")
            except Exception as e:
                dbg_log("SocketServerThread.stop", f"close failed: {e}")

class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._text_item = self._scene.addText("")
        self._text_item.setDefaultTextColor(QColor("white"))

        self._zoom = 0

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #888;
                background: #222;
            }
        """)

    def _update_placeholder_pos(self):
        if self._text_item.toPlainText():
            view_rect = self.viewport().rect()
            text_rect = self._text_item.boundingRect()
            x = max(0, (view_rect.width() - text_rect.width()) / 2)
            y = max(0, (view_rect.height() - text_rect.height()) / 2)
            scene_pos = self.mapToScene(int(x), int(y))
            self._text_item.setPos(scene_pos)

    def clear_image(self, text=""):
        self._pixmap_item.setPixmap(QPixmap())
        self._text_item.setPlainText(text)
        self._zoom = 0
        self.resetTransform()
        self._scene.setSceneRect(self.rect())
        self._update_placeholder_pos()

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.clear_image("图片加载失败")
            return False

        self._pixmap_item.setPixmap(pixmap)
        self._text_item.setPlainText("")
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_image()
        return True

    def fit_image(self):
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        self.resetTransform()
        rect = self._pixmap_item.boundingRect()
        if not rect.isNull():
            self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom = 0

    def wheelEvent(self, event):
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
            if self._zoom < 20:
                self.scale(factor, factor)
                self._zoom += 1
        else:
            factor = zoom_out_factor
            if self._zoom > -10:
                self.scale(factor, factor)
                self._zoom -= 1

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item.pixmap().isNull():
            self._update_placeholder_pos()
        elif self._zoom == 0:
            self.fit_image()


# =========================
# 主窗口
# =========================
class MainWindow(QWidget):
    def __init__(self):
        dbg_log("MainWindow.__init__", "enter")
        super().__init__()

        self.setWindowTitle("镭刻码检测识别")
        self.resize(1300, 800)

        self.image_paths = []
        self.current_index = 0
        self.output_dir = ""
        self.result_cache = {}
        self.batch_worker = None
        self.models_ready = False
        self._log_buffer = ""
        self.result_image_cache = {}

        base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
        self.temp_result_dir = os.path.join(base_dir, "_temp_result_view")
        os.makedirs(self.temp_result_dir, exist_ok=True)

        dbg_log("MainWindow.__init__", "before init_ui")
        self.init_ui()
        dbg_log("MainWindow.__init__", "after init_ui")

        dbg_log("MainWindow.__init__", "before _setup_log_redirect")
        self._setup_log_redirect()
        dbg_log("MainWindow.__init__", "after _setup_log_redirect")

        self.append_log("程序启动")

        dbg_log("MainWindow.__init__", "before start_server")
        self.start_server()
        dbg_log("MainWindow.__init__", "after start_server")

        self.status_label.setText("State: server started, loading models...")
        print("模型初始化开始")

        QApplication.processEvents()

        try:
            dbg_log("MainWindow.__init__", "before init_models")
            init_models()
            self.models_ready = True
            self.status_label.setText("State: models ready")
            self.append_log("模型初始化完成")
            dbg_log("MainWindow.__init__", "init_models success")
        except Exception as e:
            self.status_label.setText("State: model init failed")
            self.append_log(f"模型初始化失败：{e}")
            dbg_log("MainWindow.__init__", f"init_models failed: {e}")
            dbg_log("MainWindow.__init__", traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Model init failed:\n{e}")

    # =========================
    # 日志重定向
    # =========================
    def _setup_log_redirect(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self._stream = EmittingStream()
        self._stream.text_written.connect(self._handle_stream_text)

        sys.stdout = self._stream
        sys.stderr = self._stream

    def _handle_stream_text(self, text):
        if not text:
            return

        self._log_buffer += text

        while "\n" in self._log_buffer:
            line, self._log_buffer = self._log_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            # 先更新右侧 socket 状态框
            self._update_socket_info_by_log(line)

            # 再做日志精简
            simple_line = self._simplify_log_line(line)
            if simple_line:
                self.append_log(simple_line)

    def _simplify_log_line(self, line):
        """
        只保留关键日志，把冗余日志过滤掉
        """
        mapping = {
            "APP_START": "程序启动",
            "MODEL_INIT_START": "模型初始化开始",
            "MODEL_INIT_DONE": "模型初始化完成",
        }

        if line in mapping:
            return mapping[line]

        # 这些是我们想保留的普通日志
        keep_prefixes = [
            "Socket 线程已启动",
            "已选择单张图片：",
            "已选择输入文件夹：",
            "已选择输出文件夹：",
            "单张识别开始：",
            "单张识别完成：",
            "单张识别失败：",
            "批量识别开始",
            "批量处理中：",
            "批量结果：",
            "批量识别完成",
            "批量识别已暂停",
            "批量识别继续",
            "客户端已连接",
            "收到路径：",
            "Socket回复：",
            "模型初始化失败：",
            "Socket绑定失败：",
            "图片加载失败：",
            "文件夹中没有图片：",
            "结果图已保存：",
        ]

        for prefix in keep_prefixes:
            if line.startswith(prefix):
                return line

        # 原始 socket / 内部日志做中文简化
        if line.startswith("CLIENT_CONNECTED"):
            return "客户端已连接"
        if line.startswith("CLIENT_DISCONNECTED"):
            return "客户端已断开连接"

        if line.startswith("RX_RAW_PATH:"):
            raw_part = line.replace("RX_RAW_PATH:", "", 1).strip()
            return f"收到路径：{raw_part}"

        if line.startswith("SOCKET_REPLY:"):
            reply = line.replace("SOCKET_REPLY:", "", 1).strip()
            return f"Socket回复：{reply}"

        if line.startswith("SERVER_BIND_ERROR:"):
            err = line.replace("SERVER_BIND_ERROR:", "", 1).strip()
            return f"Socket绑定失败：{err}"

        # 这些全部忽略
        ignore_prefixes = [
            "[",  # dbg_log 那类详细日志
            "device =",  # 模型设备信息
            "loading parseq",  # 模型加载细节
            "parseq loaded",
            "loading yolo",
            "yolo loaded",
            "SERVER_READY",
            "RX_FIXED_PATH:",
            "RECOGNIZE_DONE",
            "SERVER_RUNTIME_ERROR",
            "TX_RESULT:",
        ]

        for prefix in ignore_prefixes:
            if line.startswith(prefix):
                return None

        return None


    def append_log(self, text):
        if not text:
            return

        ts = datetime.now().strftime("%H:%M:%S")
        safe_text = html.escape(text)
        self.log_text.append(
            f"<span style='color:#8ab4f8;'>[{ts}]</span> "
            f"<span style='color:#dddddd;'>{safe_text}</span>"
        )
        self._update_socket_info_by_log(text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()

    def _update_socket_info_by_log(self, text):
        if text.startswith("SERVER_READY"):
            self.server_edit.setText("True")
            self.state_edit.setText("Listening")
        elif text.startswith("SERVER_BIND_ERROR"):
            self.server_edit.setText("False")
            self.state_edit.setText("BindError")
        elif text.startswith("CLIENT_CONNECTED"):
            self.server_edit.setText("True")
            self.state_edit.setText("Connected")
        elif text.startswith("CLIENT_DISCONNECTED"):
            self.state_edit.setText("Listening")
        elif text.startswith("RX_RAW_PATH:") or text.startswith("RX_FIXED_PATH:"):
            self.state_edit.setText("Processing")
        elif text.startswith("SOCKET_REPLY:"):
            payload = text.replace("SOCKET_REPLY:", "", 1).strip()
            if is_socket_error_message(payload):
                self.state_edit.setText("ErrorReply")
            else:
                self.state_edit.setText("Ready")
        elif text.startswith("SERVER_RUNTIME_ERROR"):
            self.state_edit.setText("RuntimeError")

    # =========================
    # UI
    # =========================
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # =========================
        # 主分割：左侧显示区 + 右侧控制区
        # =========================
        splitter_style = """
        QSplitter::handle {
            background-color: #666666;
            border: 1px solid #888888;
        }
        QSplitter::handle:hover {
            background-color: #888888;
        }
        """
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(12)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setCollapsible(0, False)  # 左侧不可折叠
        self.main_splitter.setCollapsible(1, False)  # 右侧不可折叠
        self.main_splitter.setOpaqueResize(True)
        self.main_splitter.setStyleSheet(splitter_style)

        # =========================
        # 左侧：上图片，下结果+日志
        # =========================
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setHandleWidth(10)
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.setStyleSheet(splitter_style)
        self.left_splitter.setMinimumWidth(980)
        self.left_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 给 splitter 手柄加明显样式
        splitter_style = """
        QSplitter::handle {
            background-color: #5a5a5a;
            border: 1px solid #777;
        }
        QSplitter::handle:hover {
            background-color: #7a7a7a;
        }
        """
        self.main_splitter.setStyleSheet(splitter_style)
        self.left_splitter.setStyleSheet(splitter_style)

        # =========================
        # 图片显示区
        # 三列：上一张 | 图片展示区域 | 下一张
        # 中间图片区域稍微小一点
        # =========================
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(6)

        # 整体三列布局
        image_row_layout = QHBoxLayout()
        image_row_layout.setContentsMargins(0, 0, 0, 0)
        image_row_layout.setSpacing(16)

        # ---------- 第一列：上一张 ----------
        left_btn_widget = QWidget()
        left_btn_widget.setFixedWidth(130)
        left_btn_layout = QVBoxLayout(left_btn_widget)
        left_btn_layout.setContentsMargins(0, 0, 0, 0)
        left_btn_layout.addStretch()

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setFixedSize(100, 44)
        self.prev_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_btn.clicked.connect(self.show_prev_image)
        left_btn_layout.addWidget(self.prev_btn, alignment=Qt.AlignCenter)

        left_btn_layout.addStretch()

        # ---------- 第二列：中间图片展示区域 ----------
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        image_box = QFrame()
        image_box.setFrameShape(QFrame.StyledPanel)
        image_box.setMinimumWidth(780)  # 最小宽度
        image_box.setMaximumWidth(980)  # 稍微小一点，不占满整块左侧
        image_box.setMinimumHeight(520)
        image_box.setStyleSheet("""
            QFrame {
                border: 1px solid #666;
                background-color: #1f1f1f;
            }
        """)

        image_box_layout = QVBoxLayout(image_box)
        image_box_layout.setContentsMargins(8, 8, 8, 8)
        image_box_layout.setSpacing(6)

        self.image_view = ZoomableImageView()
        self.image_view.setMinimumWidth(700)
        self.image_view.setMinimumHeight(420)
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_view.clear_image("请先选择输入图片或输入文件夹")

        self.page_label = QLabel("0 / 0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("font-size: 16px; color: white; border: none;")

        image_box_layout.addWidget(self.image_view, 1)
        image_box_layout.addWidget(self.page_label, 0)

        # 中间列居中放置图片框
        center_inner_layout = QHBoxLayout()
        center_inner_layout.setContentsMargins(0, 0, 0, 0)
        center_inner_layout.addStretch()
        center_inner_layout.addWidget(image_box, 0)
        center_inner_layout.addStretch()

        center_layout.addLayout(center_inner_layout, 1)

        # ---------- 第三列：下一张 ----------
        right_btn_widget = QWidget()
        right_btn_widget.setFixedWidth(130)
        right_btn_layout = QVBoxLayout(right_btn_widget)
        right_btn_layout.setContentsMargins(0, 0, 0, 0)
        right_btn_layout.addStretch()

        self.next_btn = QPushButton("下一张")
        self.next_btn.setFixedSize(100, 44)
        self.next_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_btn.clicked.connect(self.show_next_image)
        right_btn_layout.addWidget(self.next_btn, alignment=Qt.AlignCenter)

        right_btn_layout.addStretch()

        # ---------- 三列拼接 ----------
        image_row_layout.addWidget(left_btn_widget, 0)
        image_row_layout.addWidget(center_widget, 1)
        image_row_layout.addWidget(right_btn_widget, 0)

        image_layout.addLayout(image_row_layout, 1)

        # =========================
        # 下方左右分割：结果区 + 日志区
        # =========================
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter.setHandleWidth(10)
        self.bottom_splitter.setChildrenCollapsible(False)
        self.bottom_splitter.setStyleSheet(splitter_style)

        # ---------- 结果区 ----------
        result_widget = QFrame()
        result_widget.setFrameShape(QFrame.StyledPanel)
        result_widget.setStyleSheet("""
            QFrame {
                border: 1px solid #666;
                background-color: #1f1f1f;
            }
        """)
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(6, 6, 6, 6)
        result_layout.setSpacing(6)

        # 结果标题行：加一个占位按钮，让它和日志区顶部高度一致
        result_title_row = QHBoxLayout()
        result_title_row.setContentsMargins(0, 0, 0, 0)

        result_title = QLabel("镭刻码识别结果")
        result_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #d0d0d0; border: none;")

        self.result_placeholder_btn = QPushButton("占位")
        self.result_placeholder_btn.setFixedSize(96, 36)
        self.result_placeholder_btn.setVisible(False)  # 只占位，不显示

        result_title_row.addWidget(result_title)
        result_title_row.addStretch()
        result_title_row.addWidget(self.result_placeholder_btn)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("这里显示镭刻码识别结果")
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-size: 24px;
                color: #f0f0f0;
                background: #2b2b2b;
                border: 1px solid #555;
            }
        """)

        result_layout.addLayout(result_title_row)
        result_layout.addWidget(self.result_text)

        # ---------- 日志区 ----------
        log_widget = QFrame()
        log_widget.setFrameShape(QFrame.StyledPanel)
        log_widget.setStyleSheet("""
            QFrame {
                border: 1px solid #666;
                background-color: #1f1f1f;
            }
        """)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(6)

        log_title_row = QHBoxLayout()
        log_title_row.setContentsMargins(0, 0, 0, 0)

        log_title = QLabel("运行日志")
        log_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #d0d0d0; border: none;")

        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.setFixedSize(96, 36)
        self.btn_clear_log.clicked.connect(self.clear_log)

        log_title_row.addWidget(log_title)
        log_title_row.addStretch()
        log_title_row.addWidget(self.btn_clear_log)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("这里显示模型初始化、Socket、识别完成等日志")
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-size: 13px;
                color: #dddddd;
                background: #1f1f1f;
                border: 1px solid #555;
                font-family: Consolas, 'Microsoft YaHei UI';
            }
        """)

        log_layout.addLayout(log_title_row)
        log_layout.addWidget(self.log_text)

        # 底部左右分割
        self.bottom_splitter.addWidget(result_widget)
        self.bottom_splitter.addWidget(log_widget)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 1)
        self.bottom_splitter.setSizes([560, 560])

        # 左侧上下分割
        self.left_splitter.addWidget(image_widget)
        self.left_splitter.addWidget(self.bottom_splitter)
        self.left_splitter.setStretchFactor(0, 4)
        self.left_splitter.setStretchFactor(1, 2)
        self.left_splitter.setSizes([700, 280])

        # =========================
        # 右侧控制区
        # =========================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        right_layout.addWidget(QLabel("输入路径："))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        right_layout.addWidget(self.input_path_edit)

        self.btn_choose_image = QPushButton("选择单张图片")
        self.btn_choose_folder = QPushButton("选择输入文件夹")
        self.btn_choose_image.clicked.connect(self.choose_single_image)
        self.btn_choose_folder.clicked.connect(self.choose_input_folder)
        right_layout.addWidget(self.btn_choose_image)
        right_layout.addWidget(self.btn_choose_folder)

        right_layout.addSpacing(10)

        right_layout.addWidget(QLabel("输出文件夹："))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        right_layout.addWidget(self.output_path_edit)

        self.btn_choose_output = QPushButton("选择输出文件夹")
        self.btn_choose_output.clicked.connect(self.choose_output_folder)
        right_layout.addWidget(self.btn_choose_output)

        right_layout.addSpacing(20)

        self.btn_single = QPushButton("单张识别")
        self.btn_batch = QPushButton("批量识别")
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setEnabled(False)

        self.btn_single.setStyleSheet("font-size: 22px; font-weight: 700; background-color: #2F80ED;")
        self.btn_batch.setStyleSheet("font-size: 22px; font-weight: 700; background-color: #2F80ED;")
        self.btn_pause.setStyleSheet("font-size: 22px; font-weight: 700; background-color: #2F80ED;")

        self.btn_single.clicked.connect(self.single_recognize)
        self.btn_batch.clicked.connect(self.batch_recognize)
        self.btn_pause.clicked.connect(self.toggle_pause_batch)

        right_layout.addWidget(self.btn_single)
        right_layout.addWidget(self.btn_batch)
        right_layout.addWidget(self.btn_pause)

        right_layout.addSpacing(20)

        self.status_label = QLabel("状态：未开始")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 20px; font-weight: 600; color: green;")
        right_layout.addWidget(self.status_label)

        right_layout.addStretch()

        self.socket_label = QLabel("Socket")
        self.socket_label.setStyleSheet("font-size: 20px; font-weight: 600; color: white;")
        right_layout.addWidget(self.socket_label)

        self.ip_label = QLabel("IP:")
        self.ip_edit = QLineEdit("127.0.0.1")
        self.port_label = QLabel("Port:")
        self.port_edit = QLineEdit("5000")
        self.server_label = QLabel("Server:")
        self.server_edit = QLineEdit("False")
        self.state_label = QLabel("State:")
        self.state_edit = QLineEdit("False")

        self.ip_label.setStyleSheet("font-size: 18px;")
        self.port_label.setStyleSheet("font-size: 18px;")
        self.server_label.setStyleSheet("font-size: 18px;")
        self.state_label.setStyleSheet("font-size: 18px;")
        self.ip_edit.setStyleSheet("font-size: 18px;")
        self.port_edit.setStyleSheet("font-size: 18px;")
        self.server_edit.setStyleSheet("font-size: 18px;")
        self.state_edit.setStyleSheet("font-size: 18px;")

        right_layout.addWidget(self.ip_label)
        right_layout.addWidget(self.ip_edit)
        right_layout.addWidget(self.port_label)
        right_layout.addWidget(self.port_edit)
        right_layout.addWidget(self.server_label)
        right_layout.addWidget(self.server_edit)
        right_layout.addWidget(self.state_label)
        right_layout.addWidget(self.state_edit)

        # =========================
        # 主布局
        # =========================
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([1350, 420])

        main_layout.addWidget(self.main_splitter)

        self.update_page_label()
        self.update_nav_buttons()
        self.result_text.setHtml("<div style='font-size:24px; color:#666666;'>None</div>")


    def start_server(self):
        dbg_log("start_server", "enter")
        self.server_thread = SocketServerThread("127.0.0.1", 5000, self.process_image)
        dbg_log("start_server", f"server_thread created id={id(self.server_thread)}")
        self.server_thread.start()
        dbg_log("start_server", "server_thread.start called")
        self.status_label.setText("状态：服务器正在运行...")
        self.append_log("Socket 线程已启动")

    def process_image(self, image_path):
        dbg_log("process_image", f"enter raw={repr(image_path)}")

        if not self.models_ready:
            msg = "模型尚未初始化完成"
            print(f"SOCKET_REPLY: {msg}")
            dbg_log("process_image", "models not ready")
            return msg

        ok, payload = validate_socket_image_path(image_path)
        if not ok:
            msg = socket_msg_to_cn(payload)
            print(f"SOCKET_REPLY: {msg}")
            dbg_log("process_image", f"validate failed -> {payload}, raw={repr(image_path)}")
            return msg

        fixed_path = payload
        dbg_log("process_image", f"validated path={repr(fixed_path)}")

        try:
            result = detect_and_recognize_image(fixed_path)
            dbg_log(
                "process_image",
                f"detect finished, has_result={result.get('has_result')}, error={result.get('error')}"
            )
        except Exception as e:
            msg = "程序内部错误"
            print(f"SOCKET_REPLY: {msg}")
            dbg_log("process_image", f"detect exception: {e}")
            dbg_log("process_image", traceback.format_exc())
            return msg

        err = str(result.get("error", "")).strip()
        if err:
            err_upper = err.upper()
            if err_upper.startswith("IMAGE_OPEN_FAILED"):
                msg = "图片打开失败"
                print(f"SOCKET_REPLY: {msg}")
                dbg_log("process_image", f"image open failed: {err}")
                return msg

            msg = "程序内部错误"
            print(f"SOCKET_REPLY: {msg}")
            dbg_log("process_image", f"unknown detection error: {err}")
            return msg

        if result.get("has_result"):
            recognized_text = str(result.get("best_text", "")).strip()
            recognized_text = "".join(ch for ch in recognized_text if ord(ch) < 128)

            if recognized_text:
                print(f"RECOGNIZE_DONE {os.path.basename(fixed_path)} -> {recognized_text}")
                dbg_log("process_image", f"recognized_text={recognized_text}")
                return recognized_text

            msg = "识别结果为空"
            print(f"SOCKET_REPLY: {msg}")
            dbg_log("process_image", "recognized text empty after ascii filter")
            return msg

        msg = "未检测到镭刻码"
        print(f"SOCKET_REPLY: {msg}")
        dbg_log("process_image", f"no detection: {os.path.basename(fixed_path)}")
        return msg

    # =========================
    # 选择输入 / 输出
    # =========================
    def choose_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单张图片",
            "",
            "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not file_path:
            return

        self.image_paths = [file_path]
        self.current_index = 0
        self.input_path_edit.setText(file_path)
        self.show_current_image()
        self.show_result_for_current_image()

        self.btn_batch.setEnabled(False)
        self.btn_single.setEnabled(True)

        self.append_log(f"已选择单张图片：{file_path}")

    def choose_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹", "")
        if not folder:
            return

        exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        paths = []
        for name in os.listdir(folder):
            full_path = os.path.join(folder, name)
            if os.path.isfile(full_path) and os.path.splitext(name)[1].lower() in exts:
                paths.append(full_path)

        paths.sort()

        if not paths:
            QMessageBox.warning(self, "提示", "该文件夹中没有图片文件")
            self.append_log(f"文件夹中没有图片：{folder}")
            return

        self.image_paths = paths
        self.current_index = 0
        self.input_path_edit.setText(folder)
        self.show_current_image()
        self.show_result_for_current_image()

        self.btn_batch.setEnabled(True)
        self.btn_single.setEnabled(True)

        self.append_log(f"已选择输入文件夹：{folder}，图片数量：{len(paths)}")

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", "")
        if not folder:
            return

        self.output_dir = folder
        self.output_path_edit.setText(folder)
        self.append_log(f"已选择输出文件夹：{folder}")

    # =========================
    # 图片显示
    # =========================
    def _make_temp_result_image_path(self, image_path):
        """
        根据原图路径生成一个稳定的临时结果图路径
        """
        md5 = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:10]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.temp_result_dir, f"{base_name}_{md5}_result.jpg")

    def update_result_view_image(self, image_path, result):
        """
        根据识别结果生成带框结果图，并缓存起来用于界面显示
        """
        if result and result.get("recognized_items"):
            temp_result_path = self._make_temp_result_image_path(image_path)
            save_result_image(image_path, temp_result_path, result)
            self.result_image_cache[image_path] = temp_result_path
        else:
            # 没检测到结果时，仍显示原图
            self.result_image_cache.pop(image_path, None)

    def show_current_image(self):
        if not self.image_paths:
            self.image_view.clear_image("请先选择输入图片或输入文件夹")
            self.update_page_label()
            self.update_nav_buttons()
            return

        img_path = self.image_paths[self.current_index]

        # 如果已有识别结果图，就显示识别结果图；否则显示原图
        display_path = self.result_image_cache.get(img_path, img_path)

        ok = self.image_view.set_image(display_path)

        if not ok:
            self.append_log(f"图片加载失败：{display_path}")

        self.update_page_label()
        self.update_nav_buttons()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def show_prev_image(self):
        if not self.image_paths:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.show_result_for_current_image()

    def show_next_image(self):
        if not self.image_paths:
            return
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
            self.show_result_for_current_image()
        else:
            self.current_index = 0
            self.show_current_image()
            self.show_result_for_current_image()

    def update_page_label(self):
        total = len(self.image_paths)
        if total == 0:
            self.page_label.setText("0 / 0")
        else:
            self.page_label.setText(f"{self.current_index + 1} / {total}")

    def update_nav_buttons(self):
        total = len(self.image_paths)
        self.prev_btn.setEnabled(total > 1 and self.current_index > 0)
        self.next_btn.setEnabled(total > 1 and self.current_index < total - 1)

    # =========================
    # 结果显示
    # =========================
    def format_result_html(self, result):
        if not result["has_result"]:
            return "<div style='font-size:24px; color:#666666;'>None</div>"

        item = result["recognized_items"][0]
        code_text = html.escape(item["text"])
        return f"<div style='font-size:30px; color:red; font-weight:700;'>{code_text}</div>"

    def show_result_for_current_image(self):
        if not self.image_paths:
            self.result_text.clear()
            return

        img_path = self.image_paths[self.current_index]
        result = self.result_cache.get(img_path)

        if result is None:
            self.result_text.setHtml("<div style='font-size:24px; color:#666666;'>None</div>")
        else:
            self.result_text.setHtml(self.format_result_html(result))

    # =========================
    # 单张识别
    # =========================
    def single_recognize(self):
        if not self.image_paths:
            QMessageBox.warning(self, "提示", "请先选择输入图片或输入文件夹")
            self.append_log("单张识别取消：未选择图片")
            return

        img_path = self.image_paths[self.current_index]
        self.status_label.setText("状态：正在进行单张识别...")
        self.append_log(f"单张识别开始：{img_path}")

        try:
            result = detect_and_recognize_image(img_path)
            self.result_cache[img_path] = result

            # 更新左侧文字结果
            self.result_text.setHtml(self.format_result_html(result))

            # 生成并显示带框结果图
            self.update_result_view_image(img_path, result)
            self.show_current_image()

            # 如果用户选择了输出目录，额外保存一份结果图
            if self.output_dir:
                save_name = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
                save_path = os.path.join(self.output_dir, save_name)
                save_result_image(img_path, save_path, result)
                self.append_log(f"结果图已保存：{save_path}")

            err = str(result.get("error", "")).strip()
            if err.upper().startswith("IMAGE_OPEN_FAILED"):
                self.append_log("单张识别失败：图片打开失败")
            elif result["has_result"]:
                self.append_log(f"单张识别完成：{result['best_text']}")
            else:
                self.append_log("单张识别完成：未检测到镭刻码")

            self.status_label.setText("状态：单张识别完成")

        except Exception as e:
            self.status_label.setText("状态：单张识别失败")
            self.append_log(f"单张识别失败：{e}")
            QMessageBox.critical(self, "错误", f"单张识别失败：\n{e}")

    # =========================
    # 批量识别
    # =========================
    def batch_recognize(self):
        if not self.image_paths:
            QMessageBox.warning(self, "提示", "请先选择输入图片或输入文件夹")
            self.append_log("批量识别取消：未选择图片")
            return

        if len(self.image_paths) == 1:
            reply = QMessageBox.question(
                self,
                "提示",
                "当前只有一张图片，是否继续批量识别？"
            )
            if reply != QMessageBox.Yes:
                self.append_log("批量识别已取消")
                return

        if self.batch_worker is not None and self.batch_worker.isRunning():
            QMessageBox.information(self, "提示", "批量识别已经在进行中")
            self.append_log("批量识别已在进行中")
            return

        self.batch_worker = BatchWorker(self.image_paths, self.output_dir)
        self.batch_worker.result_signal.connect(self.on_batch_result)
        self.batch_worker.status_signal.connect(self.on_batch_status)
        self.batch_worker.finished_signal.connect(self.on_batch_finished)

        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("暂停")
        self.status_label.setText("状态：开始批量识别")
        self.append_log(f"批量识别开始，数量：{len(self.image_paths)}")
        self.batch_worker.start()

    def on_batch_result(self, img_path, result):
        self.result_cache[img_path] = result

        # 更新结果图缓存
        self.update_result_view_image(img_path, result)

        if img_path in self.image_paths:
            self.current_index = self.image_paths.index(img_path)
            self.show_current_image()
            self.result_text.setHtml(self.format_result_html(result))

        err = str(result.get("error", "")).strip()
        if err.upper().startswith("IMAGE_OPEN_FAILED"):
            self.append_log(f"批量结果：{os.path.basename(img_path)} -> 图片打开失败")
        elif result["has_result"]:
            self.append_log(f"批量结果：{os.path.basename(img_path)} -> {result['best_text']}")
        else:
            self.append_log(f"批量结果：{os.path.basename(img_path)} -> 未检测到镭刻码")

    def on_batch_status(self, text):
        self.status_label.setText("状态：" + text)
        self.append_log(text)

    def on_batch_finished(self):
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.status_label.setText("状态：批量识别完成")
        self.append_log("批量识别完成")

    def toggle_pause_batch(self):
        if self.batch_worker is None or not self.batch_worker.isRunning():
            return

        self.batch_worker.toggle_pause()

        if self.batch_worker._paused:
            self.btn_pause.setText("继续")
            self.status_label.setText("状态：批量识别已暂停")
            self.append_log("批量识别已暂停")
        else:
            self.btn_pause.setText("暂停")
            self.status_label.setText("状态：批量识别继续中")
            self.append_log("批量识别继续")

    def closeEvent(self, event):
        dbg_log("closeEvent", "enter")
        try:
            if self.batch_worker is not None and self.batch_worker.isRunning():
                dbg_log("closeEvent", "stopping batch_worker")
                self.batch_worker.stop()
                self.batch_worker.wait(1000)
                dbg_log("closeEvent", "batch_worker stopped")
        except Exception as e:
            dbg_log("closeEvent", f"batch_worker stop failed: {e}")

        try:
            if hasattr(self, "server_thread") and self.server_thread is not None:
                dbg_log("closeEvent", "stopping server_thread")
                self.server_thread.stop()
                dbg_log("closeEvent", "server_thread stopped")
        except Exception as e:
            dbg_log("closeEvent", f"server_thread stop failed: {e}")

        try:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            dbg_log("closeEvent", "stdout/stderr restored")
        except Exception as e:
            dbg_log("closeEvent", f"restore stdout/stderr failed: {e}")

        dbg_log("closeEvent", "super().closeEvent before")
        super().closeEvent(event)
        dbg_log("closeEvent", "exit")


if __name__ == "__main__":
    dbg_log("__main__", f"program entry, argv={sys.argv}")
    dbg_log("__main__", f"executable={sys.executable}")
    dbg_log("__main__", f"cwd={os.getcwd()}")

    app = QApplication(sys.argv)
    dbg_log("__main__", "QApplication created")

    window = MainWindow()
    dbg_log("__main__", "MainWindow created")

    window.show()
    dbg_log("__main__", "MainWindow shown")

    ret = app.exec()
    dbg_log("__main__", f"app.exec finished, ret={ret}")
    sys.exit(ret)