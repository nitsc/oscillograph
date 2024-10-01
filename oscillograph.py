import numpy as np
import matplotlib.pyplot as plt
import pyaudio

# 设置参数
CHUNK = 1024
RATE = 44100

# 初始化 PyAudio
p = pyaudio.PyAudio()

# 打开流
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# 创建图形
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line, = ax1.plot(np.zeros(CHUNK), linewidth=0.5)
ax1.set_ylim(-32768, 32767)
ax1.set_title("实时示波器")
ax1.set_xlabel("样本")
ax1.set_ylabel("幅度")
plt.grid()

freq_text = ax2.text(0.5, 0.5, '', fontsize=20, ha='center', va='center')
ax2.set_title("实时频率和响度")
ax2.axis('off')

def update_plot(data):
    line.set_ydata(data)
    plt.draw()
    plt.pause(0.01)

def compute_frequency_and_loudness(data):
    # 计算FFT
    fft_data = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data), 1/RATE)
    
    # 计算响度（以dB表示）
    loudness = 20 * np.log10(np.sqrt(np.mean(data**2)) / 1)  # 使用1作为参考幅度
    
    # 找到最大频率
    peak_freq = np.abs(freq[np.argmax(np.abs(fft_data))])
    return peak_freq, loudness

print("开始实时录音...")

try:
    while True:
        # 读取音频数据
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # 更新图形
        update_plot(audio_data)

        # 计算频率和响度
        peak_freq, loudness = compute_frequency_and_loudness(audio_data)
        freq_text.set_text(f"频率: {peak_freq:.2f} Hz\n响度: {loudness:.2f}")

except KeyboardInterrupt:
    print("停止录音...")

# 清理
stream.stop_stream()
stream.close()
p.terminate()
plt.ioff()
plt.show()
