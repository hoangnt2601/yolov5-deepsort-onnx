import cv2
import numpy as np
import onnxruntime as ort
import torch
from tracker import deep_sort

from utils import mns, preprocess
from tracker.deep_sort import DeepSort


if __name__ == "__main__":
	cap = cv2.VideoCapture("/media/mdt/Data2/video/sgcoop/2554.mp4")

	# config model
	img_size = 640
	input_names = ["input0"]
	output_names = ["output"]
	conf_thres = 0.4
	iou_thres = 0.5
	# load model
	session = ort.InferenceSession("weights/yolov5s.onnx")

	width = 1920
	height = 1080

	ds = DeepSort("./weights/original_ckpt.onnx")
	# ds = DeepSort("./weights/original_ckpt.t7")
	out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (width, height))

	while True:
		frame = cap.read()[1]
		if frame is None:
			break

		frame = cv2.resize(frame, (width, height))

		H, W = frame.shape[:2]

		# detect person
		input_, scale = preprocess(frame, img_size)
		input_ = np.expand_dims(input_, 0)

		boxes = []

		try:
			outputs = session.run(output_names, {input_names[0]: input_})
			dets = mns(
				torch.tensor(outputs[0]),
				conf_thres,
				iou_thres
			)[0]
		except Exception as e:
			print("{0}: {1}".format(type(e), e))

		# if dets is None or dets.nelement() == 0:
		# 	continue

		for det in dets:
			det = det.cpu().detach().numpy()
			x1, y1, x2, y2, score = (det / scale).astype(np.float32)
			# cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
			cx = (x1 + x2) // 2
			cy = (y1 + y2) // 2
			w = x2 - x1
			h = y2 - y1
			boxes.append([cx, cy, w, h])

		tracked_boxes = ds.update(boxes, frame)

		for bbox in tracked_boxes:
			x, y, w, h, trk_id = bbox
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
			cv2.putText(frame, str(trk_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
				   1, (255,0, 0), 2, cv2.LINE_AA)

		out.write(frame)

		# cv2.imshow("f", cv2.resize(frame, (1280, 720)))
		# if cv2.waitKey(20) == ord("q"):
		# 	break

	cap.stop()
	cv2.destroyAllWindows()
