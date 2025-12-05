import serial
import threading
import time

class SerialMonitor:
    def __init__(self, port='COM1', baudrate=9600):
        """
        初始化串口监视器
        
        Args:
            port: 串口端口号
            baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.in_data = ""  # 存储从串口接收到的数据
        self.out_data = ""  # 存储待发送到串口的数据
        self.send_flag = False  # 控制是否发送数据的布尔变量
        self.running = False
        
    def connect(self):
        """连接到串口"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"成功连接到串口 {self.port}")
            return True
        except Exception as e:
            print(f"连接串口失败: {e}")
            return False
    
    def disconnect(self):
        """断开串口连接"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("串口连接已关闭")
    
    def read_serial(self):
        """读取串口数据线程"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.is_open:
                    # 读取串口数据
                    if self.serial_conn.in_waiting > 0:
                        data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8')
                        self.in_data += data
                        print(f"接收到数据: {data.strip()}")
                time.sleep(0.1)
            except Exception as e:
                print(f"读取串口数据出错: {e}")
    
    def write_serial(self):
        """发送数据到串口线程"""
        while self.running:
            try:
                if self.send_flag and self.out_data and self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.write(self.out_data.encode('utf-8'))
                    print(f"发送数据: {self.out_data}")
                    self.out_data = ""  # 发送后清空数据
                    self.send_flag = False  # 重置发送标志
                time.sleep(0.1)
            except Exception as e:
                print(f"发送串口数据出错: {e}")
    
    def start_monitoring(self):
        """开始监视串口"""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return False
                
        self.running = True
        
        # 启动读取和写入线程
        read_thread = threading.Thread(target=self.read_serial)
        write_thread = threading.Thread(target=self.write_serial)
        
        read_thread.daemon = True
        write_thread.daemon = True
        
        read_thread.start()
        write_thread.start()
        
        print("串口监视已启动")
        return True
    
    def stop_monitoring(self):
        """停止监视串口"""
        self.running = False
        self.disconnect()
        print("串口监视已停止")

# 示例用法
if __name__ == "__main__":
    # 创建串口监视器实例
    monitor = SerialMonitor(port='COM1', baudrate=9600)  # 根据实际情况修改端口号
    
    # 开始监视
    if monitor.start_monitoring():
        try:
            while True:
                # 模拟设置要发送的数据
                # 这里可以根据需要修改out_data的值
                user_input = input("输入要发送的数据 (输入'quit'退出): ")
                if user_input.lower() == 'quit':
                    break
                elif user_input:
                    monitor.out_data = user_input + '\n'
                    monitor.send_flag = True  # 设置发送标志为True
                    
                # 显示接收到的数据
                if monitor.in_data:
                    print(f"当前接收缓冲区内容: {monitor.in_data}")
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            monitor.stop_monitoring()
    else:
        print("无法启动串口监视")