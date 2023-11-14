Fs = 100;
T = 1/Fs;
L = 500;
Y = dlmread('f12.txt', ' ');
Y = Y(1:end-1);
t = (0:L-1)*T;

% Plot the signal
figure;
subplot(2,1,1);
plot(t, Y);
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Compute and plot the Fourier Transform
Yf = fft(Y);
subplot(2,1,2);
plot(abs(Yf));
title('Fourier Transform');
xlabel('Frequency');
ylabel('Amplitude');

% Compute and plot the single-sided amplitude spectrum
Sp = abs(Yf)*2/L;
Sp = Sp(1:L/2);
tf = (0:L/2-1)*Fs/L;

figure;
plot(tf, Sp);
title('Single-Sided Amplitude Spectrum');
xlabel('Frequency (Hz)');
ylabel('Amplitude');

% Find local maxima in the spectrum
localMax = [];
for i = 2:length(Sp)-1
    if Sp(i-1) < Sp(i) && Sp(i) > Sp(i+1)
        localMax(end+1) = tf(i);
    end
end

disp('Local Maxima:');
disp(localMax);