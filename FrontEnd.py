from PyQt5.QtWidgets import QGraphicsOpacityEffect


def widget_opacity(widget, op_value):
	op = QGraphicsOpacityEffect()
	# 設置透明度
	op.setOpacity(op_value)
	widget.setGraphicsEffect(op)


# 設定按鈕使用
def set_button_enable(button_list, op_value, bool_able):
	for button in button_list:
		widget_opacity(button, op_value)
		button.setEnabled(bool_able)

# 錯誤視窗
def err_dialog(dia, message):
	error_dialog = dia
	error_dialog.show_error(message)
	error_dialog.exec_()




