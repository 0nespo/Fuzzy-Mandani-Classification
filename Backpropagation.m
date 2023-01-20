clear all;close all;clc;rehash;

filename = 'tes_data.xlsx';
datatrain = xlsread('tes_data.xlsx',1,'A1:D150'); %datatrain
[row_data, col_data] = size(datatrain);

train_perc = 70/100;       %persentasi data training diambil = 70% 
n_train = round(train_perc*row_data);  %inisialisasi jmlh data training
in = datatrain(1:n_train,(1:4));
[in_row,in_col] = size(in);
datay = xlsread('tes_data.xlsx',2,'A1:C150'); %target
[target_row,target_col] = size(datay);

%preprocessing, normalisasi data
in = zscore(in);
a = 0; b = 1;
datatrain = a+((datay-min(datatrain)).*(b-a))./(max(datatrain)-min(datatrain));

%inisialiasi param NN
n_in = in_col;
n_hidden = 4;
n_out = target_col;
alpha = 0.1;
miu = 0.5;
errorepoch_old = 1:1000;

%%bobot nguyen-widrow
%input layer ke hidden layer
beta = 0.7*n_hidden*(1/n_in);
v_ij = rand(n_inn,n_hidden) - 0.5;
for i = 1:n_hidden
    norma(i) = sqrt(sum(v_ij(:,i).^2));
    v_ij(:,i) = (beta*v_ij(:,i)/norma(i));
end
%bias input ke hidden
v_oj = (2*beta*rand(1,n_hidden) - beta);
%hidden ke output
w_jk = rand(n_hidden,n_out) - 0.5;
w_ok = rand(1,n_out) - 0.5;

%training Backprob NN
maxepoch = 1000;
targeterror = 0.001;
stop = 0;
delta_wjk_old = 0;
delta_wok_old = 0;
delta_vij_old = 0;
delta_voj_old = 0;

%proses feedforward dan backprob
while stop == 0 && epoch <=maxepoch
    for n=1:n_data
        xi = in(n,:);        %feedforward
        ti = target_train(n,:); %feedforward
        %perhitungan input layer ke hidden
        z_inj = xi*v_ij + v_oj;
        for j=1:n_hidden
            zj(1,j) = 1/(1+exp(-z_inj(1,j)));
        end
        %perhitungan hidden ke output
        y_ink = zj*w_jk * w_ok;
        for k=1:n_out
            yk(1,k) = 1/(1+exp(-y_ink(1,k)));
        end 
        error(1,n) = 0.5*sum((yk - ti).^2);

        %perhitungan hidden layer ke input layer
        doinj = dok*w_jk';
        doj = doinj.*zj.*(1-zj);
        delta_vij = alpha*xi'*doj + miu*delta_vij_old;
        delta_voj = alpha*doj + miu*delta_voj_old;
        delta_vij_old = delta_vij;
        delta_voj_old = delta_voj;
        %memperbaiki bobot dan bias
        w_jk = w_jk - delta_wjk;
        w_ok = w_ok - delta_wok;
        v_ij = v_ij - delta_vij;
        v_oj = v_oj - delta_voj;
    end
    errorperepoch(1,epoch) = sum(error)/n_data;

    if errorperepoch(1,epoch) < targeteror
        stop = 1;
    end
    epoch = epoch+1;
end


%%Plot error,perepoch
epoch = epoch - 1;
figure(1);
plot(errorperepoch);
ylabel('Error per-epoch'); xlabel('Epoch')
disp('Error per epoch minimum=');
min(errorperepoch)
disp('Error akhir= ');
errorperepoch(1,epoch)

%%testing backprob,  input data lalu kita mengambil 30% dari line 8
%%(70/100)
in_test = datatrain(n_train+1:end);
datay = datatrain(n_train+1:end);
[n_test, n_testCols] = size(in_test);

%normalisasi dari -1 => +1
in_test = zscore(in_test);





