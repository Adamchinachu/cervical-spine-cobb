import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from pathlib import Path
import sys

# 导入自定义模块
from new_script import process_folder, main

class VertebralAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("颈椎角度分析工具")
        self.root.geometry("500x400")
        # 设置最小窗口大小
        self.root.minsize(600, 800)
        
        # 模型路径（固定）
        self.c2c7_model_path = "pyd\\c2_c7.pt"
        self.leftright_model_path = "pyd\\leftright.pt"
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主标题
        title_label = tk.Label(self.root, text="颈椎角度分析工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 模型状态检查
        self.check_model_status()
        
        # 分隔线
        separator1 = ttk.Separator(self.root, orient='horizontal')
        separator1.pack(fill='x', padx=20, pady=10)
        
        # 单张图片处理区域
        single_frame = tk.LabelFrame(self.root, text="单张图片处理", font=("Arial", 12))
        single_frame.pack(fill='x', padx=20, pady=10)
        
        self.selected_image_label = tk.Label(single_frame, text="未选择图片", fg="gray")
        self.selected_image_label.pack(pady=5)
        
        select_image_btn = tk.Button(single_frame, text="选择图片", command=self.select_single_image, 
                                   bg="#4CAF50", fg="white", font=("Arial", 10))
        select_image_btn.pack(pady=5)
        
        process_single_btn = tk.Button(single_frame, text="处理单张图片", command=self.process_single, 
                                     bg="#2196F3", fg="white", font=("Arial", 10))
        process_single_btn.pack(pady=5)
        
        # 分隔线
        separator2 = ttk.Separator(self.root, orient='horizontal')
        separator2.pack(fill='x', padx=20, pady=10)
        
        # 批量处理区域
        batch_frame = tk.LabelFrame(self.root, text="批量处理", font=("Arial", 12))
        batch_frame.pack(fill='x', padx=20, pady=10)
        
        self.selected_folder_label = tk.Label(batch_frame, text="未选择文件夹", fg="gray")
        self.selected_folder_label.pack(pady=5)
        
        select_folder_btn = tk.Button(batch_frame, text="选择文件夹", command=self.select_folder, 
                                    bg="#4CAF50", fg="white", font=("Arial", 10))
        select_folder_btn.pack(pady=5)
        
        # Excel文件名输入
        excel_frame = tk.Frame(batch_frame)
        excel_frame.pack(pady=5)
        
        tk.Label(excel_frame, text="Excel文件名:", font=("Arial", 10)).pack(side='left')
        self.excel_name_entry = tk.Entry(excel_frame, width=20, font=("Arial", 10))
        self.excel_name_entry.insert(0, "results.xlsx")
        self.excel_name_entry.pack(side='left', padx=5)
        
        process_batch_btn = tk.Button(batch_frame, text="批量处理", command=self.process_batch, 
                                    bg="#FF9800", fg="white", font=("Arial", 10))
        process_batch_btn.pack(pady=5)
        
        # 分隔线
        separator3 = ttk.Separator(self.root, orient='horizontal')
        separator3.pack(fill='x', padx=20, pady=10)
        
        # 进度显示区域
        self.progress_label = tk.Label(self.root, text="就绪", fg="blue", font=("Arial", 10))
        self.progress_label.pack(pady=5)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress_bar.pack(fill='x', padx=20, pady=5)
        
        # 状态文本框
        self.status_text = tk.Text(self.root, height=8, width=60, font=("Consolas", 9))
        self.status_text.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 滚动条
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side='right', fill='y')
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
        # 初始化变量
        self.selected_image_path = None
        self.selected_folder_path = None
        
    def check_model_status(self):
        """检查模型文件是否存在"""
        c2c7_exists = os.path.exists(self.c2c7_model_path)
        leftright_exists = os.path.exists(self.leftright_model_path)
        
        if c2c7_exists and leftright_exists:
            status_text = "✓ 模型文件检查通过"
            color = "green"
        else:
            missing_models = []
            if not c2c7_exists:
                missing_models.append("c2_c7.pt")
            if not leftright_exists:
                missing_models.append("leftright.pt")
            status_text = f"✗ 缺少模型文件: {', '.join(missing_models)}"
            color = "red"
        
        model_status_label = tk.Label(self.root, text=status_text, fg=color, font=("Arial", 10))
        model_status_label.pack(pady=5)
        
    def log_message(self, message):
        """在状态文本框中添加消息"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def select_single_image(self):
        """选择单张图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            filename = os.path.basename(file_path)
            self.selected_image_label.config(text=f"已选择: {filename}", fg="black")
            self.log_message(f"选择图片: {file_path}")
            
    def select_folder(self):
        """选择文件夹"""
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        
        if folder_path:
            self.selected_folder_path = folder_path
            folder_name = os.path.basename(folder_path)
            self.selected_folder_label.config(text=f"已选择: {folder_name}", fg="black")
            self.log_message(f"选择文件夹: {folder_path}")
            
    def process_single(self):
        """处理单张图片"""
        if not self.selected_image_path:
            messagebox.showwarning("警告", "请先选择一张图片！")
            return
            
        if not os.path.exists(self.c2c7_model_path) or not os.path.exists(self.leftright_model_path):
            messagebox.showerror("错误", "模型文件不存在！请确保c2_c7.pt和leftright.pt在当前目录下。")
            return
            
        # 在新线程中运行处理，避免界面冻结
        thread = threading.Thread(target=self._process_single_thread)
        thread.daemon = True
        thread.start()
        
    def _process_single_thread(self):
        """在线程中处理单张图片"""
        try:
            self.progress_label.config(text="正在处理单张图片...")
            self.progress_bar.start()
            self.log_message("开始处理单张图片...")
            
            # 调用处理函数 (GUI模式下显示图像窗口)
            self.log_message("处理中...结果图像将在新窗口中显示")
           
            
            result = main(
                self.selected_image_path,
                self.c2c7_model_path,
                self.leftright_model_path,
                show_image=True
            )
            
            self.progress_bar.stop()
            
            # 由于main函数可能没有返回值，我们假设如果没有异常就是成功
            self.progress_label.config(text="单张图片处理完成")
            self.log_message("单张图片处理完成！")
            self.log_message("结果已保存为 Excel/angles_new.xlsx")
            
            # 使用root.after确保在主线程中显示消息框
            self.root.after(0, lambda: messagebox.showinfo("成功", "单张图片处理完成！\n结果已保存为 Excel/angles_new.xlsx"))
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.progress_bar.stop()
            self.progress_label.config(text="处理出错")
            self.log_message(f"处理出错: {str(e)}")
            self.log_message(f"详细错误信息: {error_traceback}")
            
            # 使用root.after确保在主线程中显示错误消息框
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理出错: {str(e)}\n\n详细信息请查看日志"))
            
    def process_batch(self):
        """批量处理"""
        if not self.selected_folder_path:
            messagebox.showwarning("警告", "请先选择一个文件夹！")
            return
            
        if not os.path.exists(self.c2c7_model_path) or not os.path.exists(self.leftright_model_path):
            messagebox.showerror("错误", "模型文件不存在！请确保c2_c7.pt和leftright.pt在当前目录下。")
            return
            
        excel_name = self.excel_name_entry.get().strip()
        if not excel_name:
            excel_name = "results.xlsx"
        elif not excel_name.endswith('.xlsx'):
            excel_name += '.xlsx'
            
        # 在新线程中运行处理，避免界面冻结
        thread = threading.Thread(target=self._process_batch_thread, args=(excel_name,))
        thread.daemon = True
        thread.start()
        
    def _process_batch_thread(self, excel_name):
        """在线程中批量处理"""
        try:
            self.progress_label.config(text="正在批量处理...")
            self.progress_bar.start()
            self.log_message("开始批量处理...")
            
            # 调用批量处理函数
            process_folder(
                self.selected_folder_path,
                self.c2c7_model_path,
                self.leftright_model_path,
                excel_name
            )
            
            self.progress_bar.stop()
            self.progress_label.config(text="批量处理完成")
            self.log_message("批量处理完成！")
            self.log_message(f"结果已保存为 Excel/{excel_name}")
            
            # 使用root.after确保在主线程中显示消息框
            self.root.after(0, lambda: messagebox.showinfo("成功", f"批量处理完成！\n结果已保存为 Excel/{excel_name}"))
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.progress_bar.stop()
            self.progress_label.config(text="处理出错")
            self.log_message(f"批量处理出错: {str(e)}")
            self.log_message(f"详细错误信息: {error_traceback}")
            
            # 使用root.after确保在主线程中显示错误消息框
            self.root.after(0, lambda: messagebox.showerror("错误", f"批量处理出错: {str(e)}\n\n详细信息请查看日志"))

def main_gui():
    root = tk.Tk()
    app = VertebralAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()
