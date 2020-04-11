% Preamble based channel estimation for OFDM systems
% References: K Vasudevan, "Coherent Detection of Turbo Coded OFDM Signals Transmitted through 
% Frequency Selective Rayleigh Fading Channels", IEEE International Conference on Signal Processing 
% Computing and Control, 26-28 Sept. 2013, Shimla.
close all
clear all
clc
%---------------- SIMULATION PARAMETERS ------------------------------------
SNR_dB = 40; % SNR per bit in dB (in logarithmic scale)
num_frames = 1*(10^2); % number of frames to be simulated
FFT_len = 1024; % length of the FFT/IFFT (#subcarriers)
chan_len = 10; % actual number of channel taps
fade_var_1D = 0.5; % 1D fade variance of the channel impulse response
preamble_len = 512; % length of the preamble (training sequence)
cp_len = chan_len-1; % length of the cyclic prefix
num_bit = 2*FFT_len; % number of data bits per OFDM frame (overall rate is 2)

% SNR parameters - overall rate is 2
SNR = 10^(0.1*SNR_dB); % SNR per bit in linear scale
noise_var_1D = 0.5*2*2*fade_var_1D*chan_len/(2*FFT_len*SNR); % 1D noise variance
%--------------------------------------------------------------------------
%                        PREAMBLE GENERATION
preamble_data = randi([0 1],1,2*preamble_len);
preamble_qpsk = 1-2*preamble_data(1:2:end)+1i*(1-2*preamble_data(2:2:end));
% Avg. power of preamble part must be equal to avg. power of data part
preamble_qpsk = sqrt(preamble_len/FFT_len)*preamble_qpsk; % (4) in paper
preamble_qpsk_ifft = ifft(preamble_qpsk);
%--------------------------------------------------------------------------
%                    Channel estimation matrix
s1_matrix = zeros(preamble_len-chan_len+1,chan_len); %(26) in paper
for i1 = 1:preamble_len-chan_len+1
s1_matrix(i1,:) = preamble_qpsk_ifft(chan_len+i1-1:-1:i1);  
end
Chan_Est_Matrix = inv(s1_matrix'*s1_matrix)*s1_matrix'; % (28) in paper
%--------------------------------------------------------------------------
C_BER = 0; % channel errors initialization
tic()
%--------------------------------------------------------------------------
for frame_cnt = 1:num_frames
%                           TRANSMITTER
%Source
data = randi([0 1],1,num_bit); % data

% QPSK mapping 
mod_sig = 1-2*data(1:2:end) + 1i*(1-2*data(2:2:end));

% IFFT operation
T_qpsk_sig = ifft(mod_sig); % T stands for time domain

% inserting cyclic prefix and preamble
T_trans_sig = [preamble_qpsk_ifft T_qpsk_sig(end-cp_len+1:end) T_qpsk_sig]; 
%--------------------------------------------------------------------------
%                            CHANNEL   
% Rayleigh channel
fade_chan = sqrt(fade_var_1D)*randn(1,chan_len) + 1i*sqrt(fade_var_1D)*randn(1,chan_len);     

% AWGN
white_noise = sqrt(noise_var_1D)*randn(1,FFT_len + cp_len + preamble_len + chan_len - 1) ...
    + 1i*sqrt(noise_var_1D)*randn(1,FFT_len + cp_len + preamble_len + chan_len - 1); 

% Channel output
Chan_Op = conv(T_trans_sig,fade_chan) + white_noise; % Chan_Op stands for channel output
%--------------------------------------------------------------------------
%                          RECEIVER 
% Channel estimation
% estimated fade channel
est_fade_chan = Chan_Est_Matrix*Chan_Op(chan_len:preamble_len).'; 
est_fade_chan = est_fade_chan.'; % now a row vector
est_freq_response = fft(est_fade_chan,FFT_len);

% discarding preamble
Chan_Op(1:preamble_len) = [];

% discarding cyclic prefix and transient samples
Chan_Op(1:cp_len) = [];
T_REC_SIG_NO_CP = Chan_Op(1:FFT_len);

% PERFORMING THE FFT
F_REC_SIG_NO_CP = fft(T_REC_SIG_NO_CP);

% ML DETECTION
QPSK_SYM = [1+1i 1-1i -1+1i -1-1i];
QPSK_SYM1 = QPSK_SYM(1)*ones(1,FFT_len);
QPSK_SYM2 = QPSK_SYM(2)*ones(1,FFT_len);
QPSK_SYM3 = QPSK_SYM(3)*ones(1,FFT_len);
QPSK_SYM4 = QPSK_SYM(4)*ones(1,FFT_len);
DIST = zeros(4,FFT_len);
DIST(1,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM1)).^2; 
DIST(2,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM2)).^2;
DIST(3,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM3)).^2;
DIST(4,:)=(abs(F_REC_SIG_NO_CP - est_freq_response.*QPSK_SYM4)).^2; 
% COMPARING EUCLIDEAN DISTANCE
[~,INDICES] = min(DIST,[],1);
% MAPPING INDICES TO QPSK SYMBOLS
DEC_QPSK_MAP_SYM = QPSK_SYM(INDICES);
% DEMAPPING QPSK SYMBOLS TO BITS
dec_data = zeros(1,num_bit);
dec_data(1:2:end) = real(DEC_QPSK_MAP_SYM)<0;
dec_data(2:2:end) = imag(DEC_QPSK_MAP_SYM)<0;
% CALCULATING BIT ERRORS IN EACH FRAME
C_BER = C_BER + nnz(data-dec_data);
end
toc()
% bit error rate
BER = C_BER/(num_bit*num_frames)
