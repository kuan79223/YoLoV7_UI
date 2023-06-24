import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtWidgets import QLabel, QTableWidgetItem, QHeaderView, QSizePolicy, QGraphicsScene, QFileDialog, \
    QAbstractItemView, QGraphicsView

from numpy import random

import FrontEnd
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import Predict_view_ui
import globals as gl
import thread_dialog


def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


def predict():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    opt.project = os.path.join(gl.PROJECT_FOLDER, 'runs/detect')
    opt.weights = gl.PREDICT_WEIGHTS
    opt.source = gl.PREDICT_SOURCE

    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)

'''預測執行緒'''
class ThreadPredict(QThread):
    SIGNAL_FINISHED = pyqtSignal()

    def __init__(self, python):
        super(ThreadPredict, self).__init__()
        self.this_python = python
        self.running = True
        self.SIGNAL_FINISHED.connect(self.finish)

    def run(self):
        self.running = True
        if self.running:
            try:
                predict()
            except Exception as e:
                print(e)
            finally:
                self.SIGNAL_FINISHED.emit()
                print('完成預測')

    def finish(self):
        self.running = False
        print('釋放執行緒')


'''撈取預測結果'''
class LoadResultTask(QThread):

    def __init__(self, python):
        super(LoadResultTask, self).__init__()
        self.this_python = python

    def run(self):
        row = 0
        for filename in os.listdir(gl.RESULT):

            pixmap = QPixmap(os.path.join(gl.RESULT, filename))
            content = f'{os.path.basename(filename)}'
            self.this_python.SIGNAL_RESULT.emit(row, pixmap, content)
            row += 1


class PredictView(QtWidgets.QWidget, Predict_view_ui.Ui_Form):
    SIGNAL_RESULT = pyqtSignal(int, object, str)

    def __init__(self, main):
        super(PredictView, self).__init__()
        self.setupUi(self)
        self.main = main
        self.scene = QGraphicsScene()

        self.SIGNAL_RESULT.connect(self.set_result_table)
        self.load_img_task = None
        self.thread_dialog = None
        self.thread_result = LoadResultTask(self)
        self.thread_predict = ThreadPredict(self)
        # graphics view 設定
        self.scrollArea.setWidgetResizable(True)  # 讓 scrollArea自適應
        self.viewResult.setMouseTracking(True)
        self.viewResult.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.viewResult.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.viewResult.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.cbb_select()
        self.btn_source.clicked.connect(self.load_source_image)
        self.btn_result.clicked.connect(self.load_result_img)
        self.btn_model.clicked.connect(self.select_weights)
        self.btn_predict.clicked.connect(self.run_predict)
        # table widget設定 -----------------------------------------------------------------
        self.tableWidget.setColumnCount(2)   # 設定欄位數量
        self.tableWidget.setColumnWidth(1, 300)  # 設定索引 1 欄位寬度
        # 固定表格視窗大小
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 選擇整列數據
        # 隱藏 標題欄與列
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.setShowGrid(False)  # 移除表格線

        self.tableWidget.itemClicked.connect(self.on_item_clicked)  # table widget 觸發事件
        # table widget設定 -----------------------------------------------------------------

        # scene 縮放
        self.btn_zoom_ori.clicked.connect(self.fit_zoom)
        self.btn_zoom_in.clicked.connect(self.zoomin)
        self.btn_zoom_out.clicked.connect(self.zoomout)
        self.zoom_factor = 1.0
        self.zoom_step = 0.25  # 縮放步伐
        self.zoom_range = [0.1, 2]  # 設定縮放的範圍，最小10% 最大200%

    def wheelEvent(self, event):
        # 計算縮放因子並限制縮放範圍
        zoom_factor = 1 + self.zoom_step * (event.angleDelta().y() / 120)
        new_zoom_factor = self.zoom_factor * zoom_factor
        print(event.angleDelta().y())
        if new_zoom_factor > self.zoom_range[1]:
            zoom_factor = self.zoom_range[1] / self.zoom_factor

        elif new_zoom_factor < self.zoom_range[0]:
            zoom_factor = self.zoom_range[0] / self.zoom_factor
        # 更新縮放比
        self.zoom_factor *= zoom_factor
        self.viewResult.scale(zoom_factor, zoom_factor)
        # 更新顯示 label 縮放比例
        self.labelScale.setText(f'{int(self.zoom_factor * 100)}%')

    def fit_zoom(self):
        self.viewResult.resetTransform()
        self.viewResult.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        self.labelScale.setText(f'{100}%')

    def zoomin(self):
        # 計算縮放因子並限制縮放範圍
        zoom_factor = 1 + self.zoom_step
        new_zoom_factor = self.zoom_factor * zoom_factor

        if new_zoom_factor > self.zoom_range[1]:
            zoom_factor = self.zoom_range[1] / self.zoom_factor
        # 更新縮放比
        self.zoom_factor *= zoom_factor
        self.viewResult.scale(zoom_factor, zoom_factor)
        # 更新label上的縮放比
        self.labelScale.setText(f"{int(self.zoom_factor * 100)}%")

    def zoomout(self):
        # 計算縮放因子並限制縮放範圍
        zoom_factor = 1 - self.zoom_step
        new_zoom_factor = self.zoom_factor * zoom_factor

        if new_zoom_factor < self.zoom_range[0]:
            zoom_factor = self.zoom_range[0] / self.zoom_factor

        # 更新縮放比
        self.zoom_factor *= zoom_factor
        self.viewResult.scale(zoom_factor, zoom_factor)

        # 更新label上的縮放比
        self.labelScale.setText(f"{int(self.zoom_factor * 100)}%")

    def getZoomFactor(self, delta):
        return self.zoom_factor ** (delta / 120)

    def load_source_image(self):

        if gl.PROJECT_FOLDER != '':

            gl.PREDICT_SOURCE = QFileDialog.getExistingDirectory(self, '選擇要預測的影像路徑', 'D:/',
                                                                 options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

            if gl.PREDICT_SOURCE:
                FrontEnd.set_button_enable([self.btn_model], 1, True)
            if gl.PREDICT_SOURCE == '':
                print("\n取消選擇")
                return

        else:
            self.thread_dialog = thread_dialog.ThreadDialog('請先選擇專案路徑')
            self.thread_dialog.start()

    '''觸發 table widget 的事件'''
    def on_item_clicked(self, item):

        self.scene.clear()  # 清空 scene , 以便下一張圖片被觸發
        row = item.row()  # 獲得列的內容
        cell_value = self.tableWidget.item(row, 1).text()

        imgname = os.path.join(gl.RESULT, cell_value)

        img = cv2.imread(imgname)
        height, width = img.shape[:2]
        qimg = QImage(bytes(img), width, height, 3 * width, QImage.Format_BGR888)
        pixmap = QPixmap(qimg)

        self.scene.addPixmap(pixmap)
        self.viewResult.setScene(self.scene)
        self.viewResult.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    # 選擇權重
    def select_weights(self):

        if gl.PROJECT_FOLDER != '':
            gl.PREDICT_WEIGHTS, _ = QFileDialog.getOpenFileName(self, '選擇預測模型', gl.PROJECT_FOLDER, 'weights file(*.pt)')

            # 有選擇權重檔，打開預測按鈕
            if gl.PREDICT_WEIGHTS:
                FrontEnd.set_button_enable([self.btn_predict], 1, True)

            if gl.PREDICT_WEIGHTS == '':
                FrontEnd.set_button_enable([self.btn_predict], 0, False)
                print("\n取消選擇")
                return

    def cbb_select(self):
        # 選擇硬體設備
        self.cbb_device_select.addItems(gl.DEVICE)
        self.cbb_device_select.activated[str].connect(self.select_device)

    # 選擇硬體 str
    def select_device(self, text):
        if text == 'CPU':
            gl.PREDICT_DEVICE = 'cpu'
        elif text == 'GPU 1顆':
            gl.PREDICT_DEVICE = '0'
        elif text == 'GPU 2顆':
            gl.PREDICT_DEVICE = '0,1'
        elif text == 'GPU 3顆':
            gl.PREDICT_DEVICE = '0,1,2'

        print(f'選擇預測模型硬體{gl.PREDICT_DEVICE}')

    '''執行預測'''
    def run_predict(self):

        if gl.PROJECT_FOLDER != '':
            if gl.PREDICT_SOURCE != '':
                self.thread_predict.start()
                self.lb_predict_param.setText(f'預測模型: {gl.PREDICT_WEIGHTS} 預測圖像: {gl.PREDICT_SOURCE} 硬體: {gl.PREDICT_DEVICE}')
                # 預測過程按鈕停用
                FrontEnd.set_button_enable([self.btn_predict], 0, False)
        else:
            self.thread_dialog = thread_dialog.ThreadDialog('尚未選擇專案路徑')

    '''選擇結果顯示在view上'''
    def load_result_img(self):
        self.scene.clear()
        if gl.PROJECT_FOLDER != '':
            # FrontEnd.set_button_enable([self.btn_result], 1, True)
            gl.RESULT = QFileDialog.getExistingDirectory(self, '選擇預測結果的路徑', gl.PROJECT_FOLDER,
                                                         options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
            if gl.RESULT != '':
                self.thread_result.start()
            else:
                print('取消選擇\n')
                return
        else:
            self.thread_dialog = thread_dialog.ThreadDialog('找不到專案路徑')
            self.thread_dialog.start()
    '''載入結果圖的執行續信號槽'''
    def set_result_table(self, row, pixmap, content):

        self.tableWidget.setRowCount(row + 1)
        self.tableWidget.setRowHeight(row, 100)  # 動態生成每個 row 高

        label1 = QLabel()
        label1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label1.setPixmap(pixmap.scaled(100, 100, aspectRatioMode=QtCore.Qt.KeepAspectRatio))  # 讓影像自適應

        self.tableWidget.setCellWidget(row, 0, label1)
        # -------------------------------------------------------------------------------
        # 以文本的方式創建在tableWidget
        item = QTableWidgetItem(content)
        self.tableWidget.setItem(row, 1, item)

        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # print(f'在 table widget增加了一筆資料{content}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = PredictView()
    form.show()
    sys.exit(app.exec_())
