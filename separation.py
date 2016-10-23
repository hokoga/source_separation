#coding:utf-8

import sys
import numpy as np
import wave
import matplotlib.pyplot as plt
import struct
from tqdm import tqdm


def stft(x, win, step):
    N = len(win)
    M = int(np.ceil(float(len(x) - N + step) / step))

    new_x = np.zeros(N + (M - 1) * step, dtype = "float64")
    new_x[:len(x)] = x
    X = np.zeros([N, M], dtype = "complex64")
    for m in xrange(M):
        start = step * m
        X[:, m] = np.fft.fft(new_x[start:start + N] * win)

    return X


def istft(x, win, step):
    N, M = x.shape
    l = (M - 1) * step + N
    X = np.zeros(l, dtype = "float64")
    X_count = np.zeros(l, dtype = "int")

    for m in xrange(M):
        start = m * step
        X[start:start+N] += np.fft.ifft(x[:, m]).real / win
        X_count[start:start+N] += 1

    X /= X_count

    return X


def NMF(X, k, num_iter):
    M, N = X.shape
    H = np.random.rand(M, k)
    U = np.random.rand(k, N)


    error = []
    eps = np.spacing(1)
    for i in tqdm(range(num_iter)):
        Y = np.dot(H, U)
        error.append(euclid_norm(X, Y))

        H *= np.dot(X, U.T) / (np.dot(H, np.dot(U, U.T)) + eps)
        U *= np.dot(H.T, X) / (np.dot(H.T, np.dot(H, U)) + eps)

    return (H, U, error)



def PSNMF(Y, F, l, num_iter, mu):
    #D = Y -(FG + HU) + mu * norm(F.T, H)
    M, N = Y.shape
    k = F.shape[1]

    G = np.random.rand(k, N)
    U = np.random.rand(l, N)
    H = np.random.rand(M, l)

    error = []
    eps = np.spacing(1)
    H = normalize(H)

    for i in tqdm(range(num_iter)):
        error.append(euclid_norm(Y, np.dot(F, G) + np.dot(H, U)))

        H *= np.dot(Y, U.T) / (np.dot(np.dot(F, G) + np.dot(H, U), U.T) +  2 * mu * np.dot(F, np.dot(F.T, H)) + eps)
        H = normalize(H)
        U *= np.dot(H.T, Y) / (np.dot(H.T, np.dot(F, G) + np.dot(H, U)) + eps)
        G *= np.dot(F.T, Y) / (np.dot(F.T, np.dot(F, G) + np.dot(H, U)) + eps)

    return (G, H, U, error)


def normalize(X):
    N = X.shape[1]

    for n in range(N):
        X[:, n] /= sum(X[:, n])

    return X

def restore_half(X, N):
    M = X.shape[1]
    new_x = np.zeros([N, M], dtype="float64")

    for i in range(N):
        if i < N/2:
            new_x[i, :] = X[i, :]
        else:
            new_x[i, :] = X[N - i - 1, :]

    return new_x


def euclid_norm(X, Y):
    d = (X ** 2 + Y ** 2 - 2 * X * Y).sum()
    return d


def write(data, fs, bit, channel, filename):
    data = [int(x * 32767.0) for x in data]
    data = struct.pack("h" * len(data), *data)
    wf = wave.open(filename, "w")
    wf.setnchannels(channel)
    wf.setsampwidth(bit / 8)
    wf.setframerate(fs)
    wf.writeframes(data)
    wf.close()


if __name__ == "__main__":
    args = sys.argv
    wf = wave.open(args[1])
    original_data = wf.readframes(wf.getnframes())
    original_data = np.frombuffer(original_data, dtype = "int16") / 32768.0
    fs = wf.getframerate()
    channel = wf.getnchannels()
    wf.close()

    wf = wave.open(args[2])
    teacher_data = wf.readframes(wf.getnframes())
    teacher_data = np.frombuffer(teacher_data, dtype = "int16") / 32768.0
    wf.close()

    N = 512
    step = N / 4
    win = np.hamming(N)

    original_spec = stft(original_data, win, step)
    teacher_spec = stft(teacher_data, win, step)
    half_original_spec = original_spec[:N/2, :]
    half_teacher_spec = teacher_spec[:N/2, :]

    teacher_nmf = NMF(np.abs(half_teacher_spec), 100, 10)
    F = normalize(teacher_nmf[0])

    G, H, U, err = PSNMF(np.abs(half_original_spec), F, 50, 5, 100)

    target_estimate = restore_half(np.dot(F, G), N)
    left_estimate = restore_half(np.dot(H, U), N)

    angles = np.cos(np.angle(original_spec)) + 1j * np.sin(np.angle(original_spec))
    target_signal = istft(target_estimate * angles, win, step)
    left_signal = istft(left_estimate * angles, win, step)

    target_signal /= max(target_signal)
    left_signal /= max(left_signal)

    write(target_signal, fs, 16, channel, "target.wav")
    write(left_signal, fs, 16, channel, "left.wav")
