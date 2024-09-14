%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Corso: MQE
% Autore: Stefano Blando  - 0334525 
% Problem Set II - Time Series 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% housekeeping
clear 
clc  
close all 


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  GDP analysis   %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Q1: importare dataset serie storica GDP

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["date", "value"];
opts.VariableTypes = ["datetime", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "date", "InputFormat", "yyyy-MM-dd");

% Import the data
GDPEUarea19 = readtable("C:\Users\stepb\Desktop\METODI QUANTITATIVI PER L'ECONOMIA\PROBLEM SET 2\GDP_EUarea19.csv", opts);

%% Clear temporary variables
clear opts


%% Q2: esplorazione e rappresentazione 

% data exploration 
% denominazione variabili 
date = GDPEUarea19.date;
gdp_EU = GDPEUarea19.value;

% rappresentazione grafica 
figure(1)
plot(date, gdp_EU)
title('EU area GDP value from 1995(Q1) to 2022(Q4) ')
ylabel('Millions of Euros')
xlabel ('Date')

%ACF e PACF
figure(2)
subplot(2,1,1)
autocorr(gdp_EU)
title('GDP ACF')
subplot(2,1,2)
parcorr(gdp_EU)
title('GDP PACF')
%memoria pesante, serie non stazionaria

% varianza mobile
% grandezza della finestra: 1 anno = 4 quarters
windowSize = 4; 
rollingVariance = movvar(gdp_EU, windowSize);
% grafico varianza mobile
figure(3)
plot(datetime(date), rollingVariance)
title('Rolling window variance (1 year) of Eu area GDP')
ylabel('Variance')
xlabel ('Date')
% finestre mobili presentano concentrazioni di variabilità non costanti --> la
% serie non è stazioanria 

%ADF test per stazionarietà 
[h,pValue,stat,~,reg] = adftest(gdp_EU);

if h == 0
  disp('La serie storica è non stazionaria')
else
    disp('La serie storica è stazionaria')
end

disp(['Statistiche del test: ', num2str(stat)]);
disp(['Valore p: ', num2str(pValue)]);

% trasformazioni utili
% trasformazione logaritmica
figure(4)
plot(date, gdp_EU)
title('EU area GDP value from 1995(Q1) to 2022(Q4) ')
ylabel('Millions of Euros')
xlabel ('Date')
hold on
yyaxis right
plot(date, log(gdp_EU))
ylabel('log of GDP value')

% eliminazione del trend lineare
gdp_trendL = detrend(gdp_EU,'linear');
figure(5)
subplot(2,1,1)
plot(date, gdp_EU)
title('EU area GDP value from 1995(Q1) to 2022(Q4) ')
ylabel('Million of Euros')
xlabel ('Date')
subplot(2,1,2)
plot(date, gdp_trendL)
title('Linear detrended EU area GDP')
ylabel('Million of Euros')
xlabel ('Date')

% eliminazione del trend quadratico
T = length(date);
t = (1:T)';
X = [ones(T,1) t t.^2];
b = X\gdp_EU;
trendQ = X*b;
detrendQ  = gdp_EU - trendQ;

figure(6)
subplot(3,1,1)
plot(date, gdp_EU)
title('EU area GDP value from 1995(Q1) to 2022(Q4) ')
subplot(3,1,2)
title('Quadratic trend')
plot(date,trendQ )
subplot(3,1,3)
plot(date,detrendQ)
title('Quadratic detrended EU area GDP')
ylabel('Million of Euros')
xlabel ('Date')

% HP filter
g = hpfilter(gdp_EU, 1600);
detrendHP = gdp_EU- g;

figure(7)
subplot(3,1,1)
plot(datetime(date), gdp_EU)
title('EU area GDP value from 1995(Q1) to 2022(Q4)')
ylabel('Million of Euros')
xlabel ('Date')
subplot(3,1,2)
plot(datetime(date),g )
title('HP filter ')
subplot(3,1,3)
plot(datetime(date),detrendHP)
title('HP filter detrend EU area GDP ')
ylabel('Million of Euros')
xlabel ('Date')

% first differences
firstdiff = gdp_EU(2:end) - gdp_EU(1:end-1);

figure(8)
subplot(2,1,1)
plot(date(2:end,:), gdp_EU(2:end))
title('EU area GDP value from 1995(Q1) to 2022(Q4)')
ylabel('Millions of Euros')
xlabel ('Date')
subplot(2,1,2)
plot(date(2:end,:),firstdiff )
title('First differences of EU area GDP')
ylabel('Million of Euros')
xlabel ('Date')

% per l'analisi si considera gdp growth 
GDP_growth = log(gdp_EU(2:end)) - log(gdp_EU(1:end-1));

figure(9)
subplot(2,1,1)
plot(date(2:end,:), gdp_EU(2:end))
title('EU area GDP value from 1995(Q2) to 2022(Q4) ')
xlabel ('Date')
subplot(2,1,2)
plot(date(2:end,:),GDP_growth)
title('EU area GDP growth' )
xlabel ('Date')


%% Q3a: dividere GDP EU area in 2 sottoperiodi, t0 a t-4, t-3 a T, e scegliere modello 
% tagliato 4Q (2022)
GDP_growth_until2021 = GDP_growth(1:end-4);
GDP_growth_until2021_data = date(2:end-4,:);
GDP_growth_2022 = GDP_growth(end-3:end);
GDP_growth_2022_data = date(end-3:end,:);

figure(10)
subplot(2,1,1)
plot(GDP_growth_2022_data,GDP_growth_2022)
title('GDP growth from T-3 to T')
subplot(2,1,2)
plot(GDP_growth_until2021_data,GDP_growth_until2021)
title('GDP growth from ts to T-4')

figure(11)
subplot(2,1,1)
autocorr(GDP_growth_until2021)
title('ACF GDP growth until 2021 ')
subplot(2,1,2)
parcorr(GDP_growth_until2021)
title('PACF GDP growth until 2021')

% scelta modello 
%creare base modelli
Mdl(1) = arima(1,0,0);
Mdl(2) = arima(2,0,0);
Mdl(3) = arima(3,0,0);
Mdl(4) = arima(4,0,0);
Mdl(5) = arima(0,0,1);
Mdl(6) = arima(0,0,2);
Mdl(7) = arima(0,0,3);
Mdl(8) = arima(0,0,4);
Mdl(9) = arima(1,0,1);
Mdl(10) = arima(1,0,2);
Mdl(11) = arima(1,0,3);
Mdl(12) = arima(1,0,4);
Mdl(13) = arima(2,0,1);
Mdl(14) = arima(2,0,2);
Mdl(15) = arima(2,0,3);
Mdl(16) = arima(2,0,4);
Mdl(17) = arima(3,0,1);
Mdl(18) = arima(3,0,2);
Mdl(19) = arima(3,0,3);
Mdl(20) = arima(3,0,4);
Mdl(21) = arima(4,0,1);
Mdl(22) = arima(4,0,2);
Mdl(23) = arima(4,0,3);
Mdl(24) = arima(4,0,4);

%stima modelli
numMdl = numel(Mdl);
logL = zeros(numMdl,1);    
numParam = zeros(numMdl,1);
for j = 1:numMdl
    [EstMdl,~,logL(j)] = estimate(Mdl(j),GDP_growth_until2021,'Display','off');
    results = summarize(EstMdl);
    numParam(j) = results.NumEstimatedParameters;
end

%calcolare e comparare Information criteria
[~,~,ic] = aicbic(logL,numParam,107);

%miglior modello per ogni criterio
[~,minIdx] = structfun(@min,ic);
[Mdl(minIdx).Description]';

% modelli migliori per l'analisii degli IC risultano essere 
% Mdl(5) = ARIMA (0,0,1)
% Mdl(1) = ARIMA (1,0,0)

% AR1
ModAR1=arima(1,0,0);
[EsModAR1,~,logLAR1]=estimate(ModAR1,GDP_growth_until2021);
residuiAR1=infer(EsModAR1,GDP_growth_until2021);

figure(12)
subplot(2,1,1)
autocorr(residuiAR1)
title('ACF Residui AR(1) GDP growth until 2021')
subplot(2,1,2)
parcorr(residuiAR1)
title('PACF Residui AR(1) GDP growth until 2021')

% MA1
ModMA1=arima(0,0,1);
[EsModMA1,~,logLMA1]=estimate(ModMA1,GDP_growth_until2021);
residuiMA1=infer(EsModMA1,GDP_growth_until2021);

figure(13)
subplot(2,1,1)
autocorr(residuiMA1)
title('ACF Residui MA(1) GDP growth until 2021')
subplot(2,1,2)
parcorr(residuiMA1)
title('PACF Residui MA(1) GDP growth until 2021')

%proviamo a unire AR1 e MA1 in ARMA1
ModARMA1=arima(1,0,1);
[EsModARMA1,~,logLARMA1]=estimate(ModARMA1,GDP_growth_until2021);
residuiARMA1=infer(EsModARMA1,GDP_growth_until2021);

figure(14)
subplot(2,1,1)
autocorr(residuiARMA1)
title('ACF Residui AR(1) GDP growth until 2021')
subplot(2,1,2)
parcorr(residuiARMA1)
title('PACF Residui AR(1) GDP growth until 2021')

% non abbiamo miglioramenti significativi 

%% Q3b e Q3c
%forecast AR(1)
[Yhat_AR1, MSE_AR1] = forecast(EsModAR1, 4,'Y0',GDP_growth_until2021);

%RMSFE AR1
RMSFE_AR1 = sqrt(mean((GDP_growth_2022-Yhat_AR1).^2));
disp(['RMSFE: ', num2str(RMSFE_AR1), '%']);
%mape AR1
mape_AR1 = mean((abs(GDP_growth_2022 - Yhat_AR1)./ GDP_growth_2022)) * 100;
disp(['MAPE: ', num2str(mape_AR1), '%']);
%mae AR1
mae_AR1 = mean(abs(GDP_growth_2022 - Yhat_AR1)); 
disp(['MAE: ', num2str(mae_AR1)]); 


%forecast MA(1)
[Yhat_MA1, MSE_MA1] = forecast(EsModMA1, 4,'Y0',GDP_growth_until2021);

% RMSFE MA1
RMSFE_MA1 = sqrt(mean((GDP_growth_2022-Yhat_MA1).^2));
disp(['RMSFE: ', num2str(RMSFE_MA1), '%']);
%mape
mape_MA1 = mean((abs(GDP_growth_2022 - Yhat_MA1)./ GDP_growth_2022)) * 100;
disp(['MAPE: ', num2str(mape_MA1), '%']);
%mae
mae_MA1 = mean(abs(GDP_growth_2022 - Yhat_MA1)); 
disp(['MAE: ', num2str(mae_MA1)]); 


%forecast ARMA(1)
[Yhat_ARMA1, MSE_ARMA1] = forecast(EsModARMA1, 4,'Y0',GDP_growth_until2021);

% RMSFE
RMSFE_ARMA1 = sqrt(mean((GDP_growth_2022-Yhat_ARMA1).^2));
disp(['RMSFE: ', num2str(RMSFE_ARMA1), '%']);
%mape
mape_ARMA1 = mean((abs(GDP_growth_2022 - Yhat_ARMA1)./ GDP_growth_2022)) * 100;
disp(['MAPE: ', num2str(mape_ARMA1), '%']);
%mae
mae_ARMA1 = mean(abs(GDP_growth_2022 - Yhat_ARMA1)); 
disp(['MAE: ', num2str(mae_ARMA1)]); 

% confronto parametri di errore 
nomi_modelli = {'ModAR1'; 'ModMA1'; 'ModARMA1'};
RMSFE_value = [0.0036; 0.0041; 0.0045];
MAPE_value = [-165.1001; -155.9958; -149.9521];
MAE_value = [0.0031; 0.0035; 0.0039];

tabella_valori = table(RMSFE_value, MAPE_value, MAE_value, 'RowNames', nomi_modelli);

% % si preferisce il modello AR1 che ha RMSFE, MAPE e MAE minori

% vettore previsione (osservati + forecast) e vettore MSE per il periodo di forecast
myforecast_AR1 =[GDP_growth_until2021; Yhat_AR1];
myMSE_ar1=[zeros(length(GDP_growth_until2021),1);MSE_AR1];

figure(14)
plot(GDP_growth)
hold on
plot(myforecast_AR1, 'b', 'LineWidth', 0.5);
hold on
CI = 1.96*sqrt(myMSE_ar1); % Calcolo dell'intervallo di confidenza al 95%
fill([1:length(myforecast_AR1), fliplr(1:length(myforecast_AR1))], ...
     [myforecast_AR1'+CI', fliplr(myforecast_AR1'-CI')], ...
     [0 0.2 0.8], 'FaceAlpha',0.1,'EdgeColor','k');

xlabel('Time');
ylabel('GDP growth');
title('AR(1) Forecast with 95% Confidence Interval');
legend('Observed', 'Forecast', 'Confidence interval 95%')

% previsione pseudo out-of-sample non si adatta bene ai dati, memoria di un
% AR(1) ha singolo lag informativo e rimane piatta intorno alla media,
% problema di volatilità troppo alta nel break strutturale

%% Preliminary Flash Estimate GDP - EU and euro area’ per 2023 Q1
% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["date", "value"];
opts.VariableTypes = ["datetime", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "date", "InputFormat", "dd/MM/yyyy");

% Import the data
GDPEUarea192023Q1 = readtable("C:\Users\stepb\Desktop\METODI QUANTITATIVI PER L'ECONOMIA\PROBLEM SET 2\GDP_EUarea19 - 2023Q1.csv", opts);

% denominazione variabili 
date_2023 = GDPEUarea192023Q1.date;
gdp_EU_2023 = GDPEUarea192023Q1.value;

GDP_growth_2023 = log(gdp_EU_2023(2:end)) - log(gdp_EU_2023(1:end-1));

%forecast 
[Yhat_AR1_2, MSE_AR1_2] = forecast(EsModAR1, 5,'Y0',GDP_growth_until2021);
% vettore previsione (osservati + forecast) e vettore MSE per il periodo di forecast
myforecast_AR1_2 =[GDP_growth_until2021; Yhat_AR1_2];
myMSE_ar1_2=[zeros(length(GDP_growth_until2021),1);MSE_AR1_2];

figure(99)
plot(GDP_growth_2023)
hold on
plot(myforecast_AR1_2, 'b', 'LineWidth', 0.5);
hold on
CI = 1.96*sqrt(myMSE_ar1_2); % Calcolo dell'intervallo di confidenza al 95%
fill([1:length(myforecast_AR1_2), fliplr(1:length(myforecast_AR1_2))], ...
     [myforecast_AR1_2'+CI', fliplr(myforecast_AR1_2'-CI')], ...
     [0 0.2 0.8], 'FaceAlpha',0.1,'EdgeColor','k');

xlabel('Time');
ylabel('GDP growth');
title('AR(1) Forecast with 95% Confidence Interval with 2023Q1 included');
legend('Observed', 'Forecast', 'Confidence interval 95%')

% modello sembra indicare un andamento medio molto vicino a quello osservato 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Q3d: dividere GDP EU area in 2 sottoperiodi diversi, t a t-16, t-15 a T-12, e scegliere modello 
GDP_growth_until2018 = GDP_growth(1:end-16);
GDP_growth_until2018_data = date(2:end-16,:);
GDP_growth_until2019 = GDP_growth(1:end-12);
GDP_growth_until2019_data = date(2:end-12,:);
GDP_growth_2019 = GDP_growth(end-15:end-12);
GDP_growth_2019_data = date(end-15:end-12,:);

figure(15)
subplot(2,1,1)
plot(GDP_growth_2019_data,GDP_growth_2019)
title('GDP growth from T-15 to T-13')
subplot(2,1,2)
plot(GDP_growth_until2018_data,GDP_growth_until2018)
title('GDP growth from t to T-16')

figure(16)
subplot(2,1,1)
autocorr(GDP_growth_until2018)
title('ACF GDP growth until 2018')
subplot(2,1,2)
parcorr(GDP_growth_until2018)
title('PACF GDP growth until 2018')

% scelta modello 
%creare base modelli
Mdl2(1) = arima(1,0,0);
Mdl2(2) = arima(2,0,0);
Mdl2(3) = arima(3,0,0);
Mdl2(4) = arima(4,0,0);
Mdl2(5) = arima(0,0,1);
Mdl2(6) = arima(0,0,2);
Mdl2(7) = arima(0,0,3);
Mdl2(8) = arima(0,0,4);
Mdl2(9) = arima(1,0,1);
Mdl2(10) = arima(1,0,2);
Mdl2(11) = arima(1,0,3);
Mdl2(12) = arima(1,0,4);
Mdl2(13) = arima(2,0,1);
Mdl2(14) = arima(2,0,2);
Mdl2(15) = arima(2,0,3);
Mdl2(16) = arima(2,0,4);
Mdl2(17) = arima(3,0,1);
Mdl2(18) = arima(3,0,2);
Mdl2(19) = arima(3,0,3);
Mdl2(20) = arima(3,0,4);
Mdl2(21) = arima(4,0,1);
Mdl2(22) = arima(4,0,2);
Mdl2(23) = arima(4,0,3);
Mdl2(24) = arima(4,0,4);



%stima modelli
numMd2 = numel(Mdl2);
logL2 = zeros(numMd2,1);    
numParam2 = zeros(numMd2,1);
for j2 = 1:numMd2
    [EstMd2,~,logL2(j2)] = estimate(Mdl2(j2),GDP_growth_until2018,'Display','off');
    results2 = summarize(EstMd2);
    numParam2(j2) = results2.NumEstimatedParameters;
end

%calcolare e comparare Information criteria
[~,~,ic2] = aicbic(logL2,numParam2,95);

%miglior modello per ogni criterio
[~,minIdx2] = structfun(@min,ic2);
[Mdl2(minIdx2).Description]';

% modelli migliori per l'analisii degli IC risultano essere 
% Mdl2(1) = ARIMA (1,0,0)
% Mdl2(23) = ARIMA (4,0,3)
% Mdl2(2) = ARIMA (2,0,0)


% AR! 
Mod2AR1=arima(1,0,0);
[EsMod2AR1,~,logL2AR1]=estimate(Mod2AR1,GDP_growth_until2018);
residui2AR1=infer(EsMod2AR1,GDP_growth_until2018);

figure(17)
subplot(2,1,1)
autocorr(residui2AR1)
title('ACF Residui AR(1) GDP growth until 2021')
subplot(2,1,2)
parcorr(residui2AR1)
title('PACF Residui AR(1) GDP growth until 2021')

% ARMA(4,3)
Mod2ARMA43=arima(4,0,3);
[EsMod2ARMA43,~,logL2ARMA43]=estimate(Mod2ARMA43,GDP_growth_until2018);
residui2ARMA43=infer(EsMod2ARMA43,GDP_growth_until2018);

figure(18)
subplot(2,1,1)
autocorr(residui2ARMA43)
title('ACF Residui MA(1) GDP growth until 2021')
subplot(2,1,2)
parcorr(residui2ARMA43)
title('PACF Residui MA(1) GDP growth until 2021')

%AR2
Mod2AR2=arima(2,0,0);
[EsMod2AR2,~,logL2AR2]=estimate(Mod2AR2,GDP_growth_until2018);
residui2AR2=infer(EsMod2AR2,GDP_growth_until2018);

figure(19)
subplot(2,1,1)
autocorr(residui2AR2)
title('ACF Residui AR(1) GDP growth until 2021')
subplot(2,1,2)
parcorr(residui2AR2)
title('PACF Residui AR(1) GDP growth until 2021')

%%% tutti i modelli hanno ACF e PACF soddisfacenti 


%forecast AR(1)
[Yhat2_AR1, MSE2_AR1] = forecast(EsMod2AR1, 4,'Y0',GDP_growth_until2018);

%RMSFE AR1
RMSFE2_AR1 = sqrt(mean((GDP_growth_2019-Yhat2_AR1).^2));
disp(['RMSFE: ', num2str(RMSFE2_AR1)]);
%mape AR1
mape2_AR1 = mean((abs(GDP_growth_2019 - Yhat2_AR1)./ GDP_growth_2019)) * 100;
disp(['MAPE: ', num2str(mape2_AR1)]);
%mae AR1
mae2_AR1 = mean(abs(GDP_growth_2019 - Yhat2_AR1)); 
disp(['MAE: ', num2str(mae2_AR1)]); 

%forecast AR(2)
[Yhat2_AR2, MSE2_AR2] = forecast(EsMod2AR2, 4,'Y0',GDP_growth_until2018);

%RMSFE AR2
RMSFE2_AR2 = sqrt(mean((GDP_growth_2019-Yhat2_AR2).^2));
disp(['RMSFE: ', num2str(RMSFE2_AR2)]);
%mape AR1
mape2_AR2 = mean((abs(GDP_growth_2019 - Yhat2_AR2)./ GDP_growth_2019)) * 100;
disp(['MAPE: ', num2str(mape2_AR2)]);
%mae AR1
mae2_AR2 = mean(abs(GDP_growth_2019 - Yhat2_AR1)); 
disp(['MAE: ', num2str(mae2_AR2)]); 

%forecast ARMA(4,3)
[Yhat2_ARMA43, MSE2_ARMA43] = forecast(EsMod2ARMA43, 4,'Y0',GDP_growth_until2018);

%RMSFE ARMA43
RMSFE2_ARMA43 = sqrt(mean((GDP_growth_2019-Yhat2_ARMA43).^2));
disp(['RMSFE: ', num2str(RMSFE2_ARMA43)]);
%mape ARMA43
mape2_ARMA43 = mean((abs(GDP_growth_2019 - Yhat2_ARMA43)./ GDP_growth_2019)) * 100;
disp(['MAPE: ', num2str(mape2_ARMA43)]);
%mae ARMA43
mae2_ARMA43 = mean(abs(GDP_growth_2019 - Yhat2_ARMA43)); 
disp(['MAE: ', num2str(mae2_ARMA43)]); 


%%%% confronto parametri di errore 
nomi_modelli_2 = {'Mod2AR1'; 'Mod2AR2'; 'Mod2ARMA43'};
RMSFE_value_2 = [0.0024; 0.0025; 0.0021];
MAPE_value_2 = [500.6866; 504.7409; 399.7561];
MAE_value_2 = [0.0022; 0.0022; 0.0018];

tabella_valori_2 = table(RMSFE_value_2, MAPE_value_2, MAE_value_2, 'RowNames', nomi_modelli_2);

% % si preferisce il modello ARMA43 che ha RMSFE, MAPE e MAE minori

% vettore previsione (osservati + forecast) e vettore MSE per il periodo di forecast
myforecast_ARMA43 =[GDP_growth_until2018; Yhat2_ARMA43];
myMSE_arma43=[zeros(length(GDP_growth_until2018),1);MSE2_ARMA43];

figure(20)
plot(GDP_growth_until2019)
hold on
plot(myforecast_ARMA43, 'b', 'LineWidth', 0.5);
hold on
CI = 1.96*sqrt(myMSE_arma43); % Calcolo dell'intervallo di confidenza al 95%
fill([1:length(myforecast_ARMA43), fliplr(1:length(myforecast_ARMA43))], ...
     [myforecast_ARMA43'+CI', fliplr(myforecast_ARMA43'-CI')], ...
     [0 0.2 0.8], 'FaceAlpha',0.1,'EdgeColor','k');

xlabel('Time');
ylabel('GDP growth');
title('ARMA(4,3) Forecast with 95% Confidence Interval');
legend('Observed', 'Forecast', 'Confidence interval 95%')

% previsione pseudo out-of-sample si adatta bene ai dati, la
% rimozione del break strutturale migliora notevolmente la performance del
% modello, che segue lo stesso andamento delle osservazioni


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  S&P 500 analysis %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Set path
addpath('C:\Users\stepb\Desktop\METODI QUANTITATIVI PER L''ECONOMIA\PROBLEM SET 2\util')

% s&p500 financial time serie
sp = getMarketDataViaYahoo('^GSPC', '1-Jan-2010', '14-Apr-2023', '1d');

%data exploration
sp0 = sp.AdjClose;
date0=table2array(sp(:,1));

%grafico sp0 
figure(21)
plot(date0,sp0)
title('S&P500 price index from  01/01/2010 to 14/04/2023 (daily)')
ylabel('USD')
xlabel ('Date')

autocorr(sp0)
parcorr(sp0)

%varianza mobile sp0
windowSizesp = 90; %90 days = 1 quarter
rollingVariancesp = movvar(sp0, windowSizesp);

figure(22)
plot(date0, rollingVariancesp)
title('Rolling window variance (1 quarter) of Eu area GDP')
ylabel('Variance')
xlabel ('Date')

% ADFtest per sp0
[h,pValue,stat,cValue,reg] = adftest(sp0);

if h == 0
  disp('La serie storica è non stazionaria')
else
    disp('La serie storica è stazionaria')
end

disp(['Statistiche del test: ', num2str(stat)]);
disp(['Valore p: ', num2str(pValue)]);


%from prices to returns --> trasformazione logaritmica + differenze prime
spR = 100*(log(sp0(2:end))-log(sp0(1:end-1)));
dateR = table2array(sp(2:end,1));

ts = dateR(1); % inizio 
te= dateR(end);  %fine 
%xlim([ts-10, te+10])

figure(23)
subplot(2,1,1)
plot(dateR,spR)
title('S&P500 logaritmic returns from 04/01/2010 to 13/04/2023 (daily)');
xlabel('Date')
subplot(2,1,2)
autocorr(spR)
title('ACF SPR')

%% Q4: dividere SPR in 2 sottoperiodi, t0 a t-10, t-9 a T = spR(1:end-9);
spR_sample = spR(1:end-10);
spR_sample_date = dateR(1:end-10);
spR_test = spR(end-9:end);
spR_test_date = dateR(end-9:end);

figure(24)
plot(spR_sample_date,spR_sample)
title('spR training');
xlabel('Date')

figure(25)
plot(spR_test_date,spR_test)
title('spR test');
xlabel('Date')

% problema: ultimo periodo ha picco di volatilità, probabile break
% strutturale per covid-19

% proxy volatilità = rendimenti al quadrato 
r2 = (spR-mean(spR)).^2;

figure(26)
autocorr(r2)
title('ACF quadratic returns')

figure(27)
subplot(2,1,1)
plot(dateR,r2)
title('r2')
subplot(2,1,2)
autocorr(r2)
title('ACF r2')
%memoria persistente

figure(28)
qqplot(spR)
%code pesanti ma simmetrico 

%% Q4a MODELLO MA rolling window 
%rolling window
tau1 = 20;
tau2 = 40;
tau3 = 60;
tau4 = 80;
tau5 = 100;
tau6 = 120;
tau7 = 140;
tau8 = 160;

myhist1 = histvol(sp0,tau1,0);
myhist2 = histvol(sp0,tau2,0);
myhist3 = histvol(sp0,tau3,0);
myhist4 = histvol(sp0,tau4,0);
myhist5 = histvol(sp0,tau5,0);
myhist6 = histvol(sp0,tau6,0);
myhist7 = histvol(sp0,tau7,0);
myhist8 = histvol(sp0,tau8,0);

figure(29)
plot(dateR, r2)
hold on
plot(dateR,myhist1.sigma_hat_1d.^2)
title('Rolling window tau1 estimate for spR')

figure(30)
plot(dateR, r2)
hold on
plot(dateR,myhist2.sigma_hat_1d.^2)
title('Rolling window tau2 estimate for spR')

figure(31)
plot(dateR, r2)
hold on
plot(dateR,myhist3.sigma_hat_1d.^2)
title('Rolling window tau3 estimate for spR')

figure(32)
plot(dateR, r2)
hold on
plot(dateR,myhist4.sigma_hat_1d.^2)
title('Rolling window tau4 estimate for spR')

figure(33)
plot(dateR, r2)
hold on
plot(dateR,myhist5.sigma_hat_1d.^2)
title('Rolling window tau5 estimate for spR')

figure(34)
plot(dateR, r2)
hold on
plot(dateR,myhist5.sigma_hat_1d.^2)
title('Rolling window tau5 estimate for spR')

figure(35)
plot(dateR, r2)
hold on
plot(dateR,myhist6.sigma_hat_1d.^2)
title('Rolling window tau6 estimate for spR')

figure(36)
plot(dateR, r2)
hold on
plot(dateR,myhist7.sigma_hat_1d.^2)
title('Rolling window tau7 estimate for spR')

figure(37)
plot(dateR, r2)
hold on
plot(dateR,myhist8.sigma_hat_1d.^2)
title('Rolling window tau8 estimate for spR')


%%% aumentando la finestra temporale diminuisce sempre di più l'accuratezza
%%% della memoria 

T_sp = length(dateR);
myvarconst = var(spR)*ones(T_sp,1);

figure(38)
subplot(3,3,1)
plot(dateR,myhist1.sigma_hat_1d)
title('\tau = 20')
subplot(3,3,2)
plot(dateR,myhist2.sigma_hat_1d)
title('\tau = 40')
subplot(3,3,3)
plot(dateR,myhist3.sigma_hat_1d)
title('\tau = 60')
subplot(3,3,4)
plot(dateR,myhist4.sigma_hat_1d)
title('\tau = 80')
subplot(3,3,5)
plot(dateR,myhist5.sigma_hat_1d)
title('\tau = 100')
subplot(3,3,6)
plot(dateR,myhist6.sigma_hat_1d)
title('\tau = 120')
subplot(3,3,7)
plot(dateR,myhist7.sigma_hat_1d)
title('\tau = 140')
subplot(3,3,8)
plot(dateR,myhist7.sigma_hat_1d)
title('\tau = 160')
subplot(3,3,9)
plot(dateR,myvarconst)
title('\tau = T')

% all'aumentare del tau per la finestra temporale la serie della volatilità
% è sempre più smussata, fino al punto estremo com tau=T dove è una serie
% costante -->  myvarcost = valore medio finale

%problema: non tutti gli istanti temporali hanno eguale peso sulla memoria
%della serie

%evaluation 
my_rmse_hist1 = sqrt(mean((r2-(myhist1.sigma_hat_1d).^2).^2));
my_rmse_hist2 = sqrt(mean((r2-(myhist2.sigma_hat_1d).^2).^2));
my_rmse_hist3 = sqrt(mean((r2-(myhist3.sigma_hat_1d).^2).^2));
my_rmse_hist4 = sqrt(mean((r2-(myhist4.sigma_hat_1d).^2).^2));
my_rmse_hist5 = sqrt(mean((r2-(myhist5.sigma_hat_1d).^2).^2));
my_rmse_hist6 = sqrt(mean((r2-(myhist6.sigma_hat_1d).^2).^2));
my_rmse_hist7 = sqrt(mean((r2-(myhist7.sigma_hat_1d).^2).^2));
my_rmse_hist8 = sqrt(mean((r2-(myhist8.sigma_hat_1d).^2).^2));
my_rmse_varconst = sqrt(mean((r2-myvarconst).^2));

% RMSE minore si ha con tau= 20 


%% Q4b MODELLO EWMA 
mylambda0 = 0.6;
myewma0 = ewma(sp0,mylambda0,mean(spR));

mylambda1 = 0.7;
myewma1 = ewma(sp0,mylambda1,mean(spR));

mylambda2 = 0.8;
myewma2 = ewma(sp0,mylambda2,mean(spR));

mylambda3 = 0.85;
myewma3 = ewma(sp0,mylambda3,mean(spR));

mylambda4 = 0.9;
myewma4 = ewma(sp0,mylambda4,mean(spR));

mylambda5 = 0.92;
myewma5 = ewma(sp0,mylambda5,mean(spR));

mylambda6 = 0.94;
myewma6 = ewma(sp0,mylambda6,mean(spR));

mylambda7 = 0.96;
myewma7 = ewma(sp0,mylambda7,mean(spR));

mylambda8 = 0.99;
myewma8 = ewma(sp0,mylambda8,mean(spR));


figure(39)
subplot(3,3,1)
plot(dateR,r2)
hold on
plot(dateR,myewma0.sigma_hat_1d.^2)
title('\lambda=0.6')
subplot(3,3,2)
plot(dateR,r2)
hold on
plot(dateR,myewma1.sigma_hat_1d.^2)
title('\lambda=0.7')
subplot(3,3,3)
plot(dateR,r2)
hold on
plot(dateR,myewma2.sigma_hat_1d.^2)
title('\lambda=0.8')
subplot(3,3,4)
plot(dateR,r2)
hold on
plot(date0(2:end),myewma3.sigma_hat_1d.^2)
title('\lambda=0.85')
subplot(3,3,5)
plot(dateR,r2)
hold on
plot(dateR,myewma4.sigma_hat_1d.^2)
title('\lambda=0.9')
subplot(3,3,6)
plot(dateR,r2)
hold on
plot(dateR,myewma5.sigma_hat_1d.^2)
title('\lambda=0.92')
subplot(3,3,7)
plot(dateR,r2)
hold on
plot(dateR,myewma6.sigma_hat_1d.^2)
title('\lambda=0.94')
subplot(3,3,8)
plot(dateR,r2)
hold on
plot(dateR,myewma7.sigma_hat_1d.^2)
title('\lambda=0.96')
subplot(3,3,9)
plot(dateR,r2)
hold on
plot(dateR,myewma8.sigma_hat_1d.^2)
title('\lambda=0.99')


rmse0 =  sqrt(mean((r2 - myewma0.sigma_hat_1d.^2).^2))
rmse1 =  sqrt(mean((r2 - myewma1.sigma_hat_1d.^2).^2))
rmse2 =  sqrt(mean((r2 - myewma2.sigma_hat_1d.^2).^2))
rmse3 =  sqrt(mean((r2 - myewma3.sigma_hat_1d.^2).^2))
rmse4 =  sqrt(mean((r2 - myewma4.sigma_hat_1d.^2).^2))
rmse5 =  sqrt(mean((r2 - myewma5.sigma_hat_1d.^2).^2))
rmse6 =  sqrt(mean((r2 - myewma6.sigma_hat_1d.^2).^2))
rmse7 =  sqrt(mean((r2 - myewma7.sigma_hat_1d.^2).^2))
rmse8 =  sqrt(mean((r2 - myewma8.sigma_hat_1d.^2).^2))

% RMSE minore è quello di ewma1, ma potrebbe dare problemi di overfitting
% ai dati del campione di training

% forecast 10 steps recursive con ewma1 
sigmahat = NaN(10,1);
 for i = 1: 10
  sp01= sp0(1:end-10+i);
  spR1 = spR(1:end-10+i);
 myewma1 = ewma(sp01, mylambda1, mean(spR1)) ;   
 sigmahat(i) = (1-mylambda1)*spR1(end).^2+ mylambda1*myewma1.sigma_hat_1d(end).^2;
 end

figure(40)
plot(date0(end-9:end),r2(end-9:end))
hold on
plot(date0(end-9:end),sigmahat)
title('EWMA1 forecast 10 step ahead')
xlabel('Date')
ylabel('r^2')
legend('Observed', 'Forecast')

% evaluation 
myrmsfe_ewma1 = sqrt(mean((r2(end-9:end)-sigmahat).^2));

% JP Morgan lambda= 0.9

% forecast 10 steps recursive con ewma4 
sigmahat_JP = NaN(10,1);
 for i = 1: 10
  sp01= sp0(1:end-10+i);
  spR1 = spR(1:end-10+i);
 myewma4 = ewma(sp01, mylambda4, mean(spR1)) ;   
 sigmahat_JP(i) = (1-mylambda4)*spR1(end).^2+ mylambda4*myewma4.sigma_hat_1d(end).^2;
 end

figure(98)
plot(date0(end-9:end),r2(end-9:end))
hold on
plot(date0(end-9:end),sigmahat_JP)
title('EWMA4 forecast 10 step ahead')
xlabel('Date')
ylabel('r^2')
legend('Observed', 'Forecast')

% evaluation 
myrmsfe_ewma4 = sqrt(mean((r2(end-9:end)-sigmahat_JP).^2));

%% Q4c: GARCH 
spR_sample = spR(1:end-10);
spR_sample_date = dateR(1:end-10);
spR_test = spR(end-9:end);
spR_test_date = dateR(end-9:end);

spR_train = (spR(1:end-10)) - (mean(spR(1:end-10))); %demean

% ARCH (0,1)
myarch1 = garch(0,1);
myarch1_est = estimate(myarch1, spR_sample,'E0', spR_sample(1:10) );
 
varARCH1 = infer(myarch1_est, spR_sample);

%ACF arch1
figure(41)
subplot(2,1,1)
autocorr(varARCH1)
subplot(2,1,2)
plot(varARCH1)

rmse_ARCH1 =  sqrt(mean((r2(1:end-10) - varARCH1).^2));

figure(42)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH1)

% ARCH (0,2)
myarch2 = garch(0,2);
myarch2_est = estimate(myarch2, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH2 = infer(myarch2_est, spR_sample);

%ACF arch2
figure(43)
subplot(2,1,1)
autocorr(varARCH2)
subplot(2,1,2)
plot(varARCH2)

rmse_ARCH2 =  sqrt(mean((r2(1:end-10) - varARCH2).^2));

figure(44)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH2)

% ARCH (0,3)
myarch3 = garch(0,3);
myarch3_est = estimate(myarch3, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH3 = infer(myarch3_est, spR_sample);

%ACF arch3
figure(45)
subplot(2,1,1)
autocorr(varARCH3)
subplot(2,1,2)
plot(varARCH3)

rmse_ARCH3 =  sqrt(mean((r2(1:end-10) -  varARCH3).^2));

figure(46)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH3)

% ARCH (0,4)
myarch4 = garch(0,4);
myarch4_est = estimate(myarch4, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH4 = infer(myarch4_est, spR_sample);

%ACF arch4
figure(47)
subplot(2,1,1)
autocorr(varARCH4)
subplot(2,1,2)
plot(varARCH4)

rmse_ARCH4 =  sqrt(mean((r2(1:end-10) -  varARCH4).^2));

figure(48)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH4)

% ARCH (0,5)
myarch5 = garch(0,5);
myarch5_est = estimate(myarch5, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH5 = infer(myarch5_est, spR_sample);

%ACF arch5
figure(49)
subplot(2,1,1)
autocorr(varARCH5)
subplot(2,1,2)
plot(varARCH5)

rmse_ARCH5 =  sqrt(mean((r2(1:end-10) -  varARCH5).^2));

figure(50)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH5)

% ARCH (0,6)
myarch6 = garch(0,6);
myarch6_est = estimate(myarch6, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH6 = infer(myarch6_est, spR_sample);

%ACF arch6
figure(51)
subplot(2,1,1)
autocorr(varARCH6)
subplot(2,1,2)
plot(varARCH6)

rmse_ARCH6 =  sqrt(mean((r2(1:end-10) -  varARCH6).^2));

figure(52)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH6)

% ARCH (0,7)
myarch7 = garch(0,7);
myarch7_est = estimate(myarch7, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH7 = infer(myarch7_est, spR_sample);

%ACF arch7
figure(53)
subplot(2,1,1)
autocorr(varARCH7)
subplot(2,1,2)
plot(varARCH7)

rmse_ARCH7 =  sqrt(mean((r2(1:end-10) -  varARCH7).^2));

figure(54)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH7)

% ARCH (0,8)
myarch8 = garch(0,8);
myarch8_est = estimate(myarch8, spR_sample, 'E0', spR_train(1:10) );
 
varARCH8 = infer(myarch8_est, spR_sample);

%ACF arch8
figure(55)
subplot(2,1,1)
autocorr(varARCH8)
subplot(2,1,2)
plot(varARCH8)

rmse_ARCH8 =  sqrt(mean((r2(1:end-10) -  varARCH8).^2));

figure(56)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH8)

% ARCH (0,9)
myarch9 = garch(0,9);
myarch9_est = estimate(myarch9, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH9 = infer(myarch9_est, spR_sample);

%ACF arch9
figure(59)
subplot(2,1,1)
autocorr(varARCH9)
subplot(2,1,2)
plot(varARCH9)

rmse_ARCH9 =  sqrt(mean((r2(1:end-10) -  varARCH9).^2));

figure(60)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH9)

% ARCH (0,10)
myarch10 = garch(0,10);
myarch10_est = estimate(myarch10, spR_sample, 'E0', spR_sample(1:10) );
 
varARCH10 = infer(myarch10_est, spR_sample);

%ACF arch10
figure(61)
subplot(2,1,1)
autocorr(varARCH10)
subplot(2,1,2)
plot(varARCH10)

rmse_ARCH10 =  sqrt(mean((r2(1:end-10) -  varARCH10).^2));

figure(62)
plot(dateR(1:end-10), r2(1:end-10))
hold on
plot(dateR(1:end-10), varARCH10)

%ACF analysis 
figure(63)
subplot(3,3,1)
autocorr(varARCH1)
title('ACF ARCH(1)')
subplot(3,3,2)
autocorr(varARCH2)
title('ACF ARCH(2)')
subplot(3,3,3)
autocorr(varARCH3)
title('ACF ARCH(3)')
subplot(3,3,4)
autocorr(varARCH5)
title('ACF ARCH(5)')
subplot(3,3,5)
autocorr(varARCH8)
title('ACF ARCH(8)')
subplot(3,3,6)
autocorr(varARCH10)
title('ACF ARCH(10)')


%evaluation
disp(rmse_ARCH1)
disp(rmse_ARCH2)
disp(rmse_ARCH3)
disp(rmse_ARCH4)
disp(rmse_ARCH5)
disp(rmse_ARCH6)
disp(rmse_ARCH7)
disp(rmse_ARCH8)
disp(rmse_ARCH9)
disp(rmse_ARCH10)

% forecast 10 steps ahead
FvarARCH1 = forecast(myarch1_est, 10, 'Y0',spR_sample);
FvarARCH2 = forecast(myarch2_est, 10, 'Y0',spR_sample);
FvarARCH3 = forecast(myarch3_est, 10, 'Y0',spR_sample);
FvarARCH4 = forecast(myarch4_est, 10, 'Y0',spR_sample);
FvarARCH5 = forecast(myarch5_est, 10, 'Y0',spR_sample);
FvarARCH6 = forecast(myarch6_est, 10, 'Y0',spR_sample);
FvarARCH7 = forecast(myarch7_est, 10, 'Y0',spR_sample);
FvarARCH8 = forecast(myarch8_est, 10, 'Y0',spR_sample);
FvarARCH9 = forecast(myarch9_est, 10, 'Y0',spR_sample);
FvarARCH10 = forecast(myarch10_est, 10, 'Y0',spR_sample);

% forecast evaluation 
rmsfe_ARCH1F =  sqrt(mean((r2(end-9:end) -  FvarARCH1).^2))
rmsfe_ARCH2F =  sqrt(mean((r2(end-9:end) -  FvarARCH2).^2))
rmsfe_ARCH3F =  sqrt(mean((r2(end-9:end) -  FvarARCH3).^2))
rmsfe_ARCH4F =  sqrt(mean((r2(end-9:end) -  FvarARCH4).^2))
rmsfe_ARCH5F =  sqrt(mean((r2(end-9:end) -  FvarARCH5).^2))
rmsfe_ARCH6F =  sqrt(mean((r2(end-9:end) -  FvarARCH6).^2))
rmsfe_ARCH7F =  sqrt(mean((r2(end-9:end) -  FvarARCH7).^2))
rmsfe_ARCH8F =  sqrt(mean((r2(end-9:end) -  FvarARCH8).^2))
rmsfe_ARCH9F =  sqrt(mean((r2(end-9:end) -  FvarARCH9).^2))
rmsfe_ARCH10F =  sqrt(mean((r2(end-9:end) -  FvarARCH10).^2))


% modello arch8 presenta RMSFE minore 
figure(63)
plot(dateR(end-9:end), r2(end-9:end))
hold on
plot(dateR(end-9:end), FvarARCH8)
title('Forecast 10 steps ahead recursive - model ARCH(8)')
xlabel('Time')
ylabel('r^2')
legend('Proxy volatility', 'Forecast ARCH(1)')

%confronto forecast
figure(64)
plot(dateR(end-9:end), r2(end-9:end))
hold on
plot(dateR(end-9:end), FvarARCH1)
hold on
plot(dateR(end-9:end), FvarARCH2)
hold on
plot(dateR(end-9:end), FvarARCH3)
hold on
plot(dateR(end-9:end), FvarARCH4)
hold on
plot(dateR(end-9:end), FvarARCH5)
hold on
plot(dateR(end-9:end), FvarARCH6)
hold on
plot(dateR(end-9:end), FvarARCH7)
hold on
plot(dateR(end-9:end), FvarARCH8)
hold on
plot(dateR(end-9:end), FvarARCH9)
hold on
plot(dateR(end-9:end), FvarARCH10)
title('Forecast 10 steps ahead recursive - model ARCH')
xlabel('Time')
ylabel('r^2')
legend('Proxy volatility', 'Forecast ARCH(1)', 'Forecast ARCH(2)', 'Forecast ARCH(3)', 'Forecast ARCH(4)', 'Forecast ARCH(5)', ...
    'Forecast ARCH(6)', 'Forecast ARCH(7)', 'Forecast ARCH(8)', 'Forecast ARCH(9)', 'Forecast ARCH(10)')


% per migliorare ARCH servirebbe introdurre troppi parametri --> si passa
% alla versione generalizzata 


% scelta modello 
% GARCH (1,1)
mygarch1 = garch(1,1);
mygarch1_est = estimate(mygarch1, spR_sample, 'E0', spR_sample(1:10) );
varGARCH1 = infer(mygarch1_est, spR_sample);
rmse_GARCH1 =  sqrt(mean((r2(1:end-10) -  varGARCH1).^2));
disp(['RMSE: ', num2str(rmse_GARCH1)]);
%forecast garch(1,1)
F_garch1 = forecast(mygarch1_est,10,'Y0',spR_sample);
figure(65)
plot(dateR(end-9:end),r2(end-9:end))
hold on
plot(dateR(end-9:end),F_garch1)
title('Forecast 10 steps ahead recursive _ GARCH(1,1)')
legend('Observed', 'Forecast')
ylabel('r^2')
xlabel('Date')


% GARCH (2,2)
mygarch2 = garch(2,2);
mygarch2_est = estimate(mygarch2, spR_sample, 'E0', spR_sample(1:10) );
varGARCH2 = infer(mygarch2_est, spR_sample);
rmse_GARCH2 =  sqrt(mean((r2(1:end-10) -  varGARCH2).^2));
disp(['RMSE: ', num2str(rmse_GARCH2)]);
%forecast garch(2,2)
F_garch2 = forecast(mygarch2_est,10,'Y0',spR_sample);
figure(65)
plot(dateR(end-9:end),r2(end-9:end))
hold on
plot(dateR(end-9:end),F_garch2)
title('Forecast 10 steps ahead recursive _ GARCH(2,2)')
legend('Observed', 'Forecast')
ylabel('r^2')
xlabel('Date')

% GARCH (1,2)
mygarch3 = garch(1,2);
mygarch3_est = estimate(mygarch3, spR_sample, 'E0', spR_sample(1:10) );
varGARCH3 = infer(mygarch3_est, spR_sample);
rmse_GARCH3 =  sqrt(mean((r2(1:end-10) -  varGARCH3).^2));
disp(['RMSE: ', num2str(rmse_GARCH3)]);
%forecast garch(1,2)
F_garch3 = forecast(mygarch3_est,10,'Y0',spR_sample);
figure(66)
plot(dateR(end-9:end),r2(end-9:end))
hold on
plot(dateR(end-9:end),F_garch3)
title('Forecast 10 steps ahead recursive _ GARCH(1,2)')
legend('Observed', 'Forecast')
ylabel('r^2')
xlabel('Date')


% GARCH (2,1)
mygarch4 = garch(2,1);
mygarch4_est = estimate(mygarch4, spR_sample, 'E0', spR_sample(1:10) );
varGARCH4 = infer(mygarch4_est, spR_sample);
rmse_GARCH4 =  sqrt(mean((r2(1:end-10) -  varGARCH4).^2));
disp(['RMSFE: ', num2str(rmse_GARCH4)]);
%forecast garch(2,1)
F_garch4 = forecast(mygarch4_est,10,'Y0',spR_sample);
figure(67)
plot(dateR(end-9:end),r2(end-9:end))
hold on
plot(dateR(end-9:end),F_garch4)
title('Forecast 10 steps ahead recursive _ GARCH(2,1)')
legend('Observed', 'Forecast')
ylabel('r^2')
xlabel('Date')


%evaluation 
my_rmse_Fgarch1 = sqrt(mean((r2(end-9:end)-F_garch1).^2));
my_rmse_Fgarch2 = sqrt(mean((r2(end-9:end)-F_garch2).^2));
my_rmse_Fgarch3 = sqrt(mean((r2(end-9:end)-F_garch3).^2));
my_rmse_Fgarch4 = sqrt(mean((r2(end-9:end)-F_garch4).^2));

% RMSFE minore è con GARCH(1,2) 

%varianza condizionata e incondizionata
var_uncond = mygarch3_est.Constant/(1-cell2mat(mygarch3_est.GARCH(1,1))-cell2mat(mygarch3_est.ARCH(1,1))-cell2mat(mygarch3_est.ARCH(1,2)));
 
figure(67)
plot(r2)
hold on
plot(varGARCH3)
hold on
plot(var_uncond*ones(T_sp,1))
title('Variance comparison')
legend('r^2', 'Contidional variance', 'Unconditional variance')

% varianza condizionata dipende dal tempo, varianza incondizionata invece è
% sempre costante 


%% Q4d; scelta modello per forecast senza break strutturale

% divisione sotto-periodi 
spR_new = spR(1:end-944);
spR_date_new = dateR(1:end-944);
spR_sample_new = spR(1:end-954);
spR_sample_date_new = dateR(1:end-954);
spR_test_new = spR(end-953:end-944);
spR_test_date_new = dateR(end-953:end-944);
r2_new = (spR_new-mean(spR_sample_new)).^2;

figure(68)
subplot(2,1,1)
plot(spR_sample_date_new,spR_sample_new)
title('New training sample (04/01/2010 - 30/06/2019');
xlabel('Date')
ylabel('Returns')
subplot(2,1,2)
plot(spR_test_date_new,spR_test_new)
title('New testing sample (01/07/2019 - 15/05/2019');
xlabel('Date')
ylabel('Returns')



% Rolling WIndow tau=20 new sample
myhist1_new = histvol(sp0(1:end-944),tau1,0);

figure(69)
plot(spR_date_new, r2_new)
hold on
plot(spR_date_new, myhist1_new.sigma_hat_1d.^2)
title('Rolling window tau1 estimate for new sample')
xlabel('Date')
ylabel('r^2')

%evaluation 
my_rmse_hist1_new = sqrt(mean((r2_new-(myhist1_new.sigma_hat_1d).^2).^2));


% EWMA1 new sample
sigmahat_new = NaN(10,1);
 for i = 1: 10
  sp01_new= sp0(1:end-954+i);
  spR1_new = spR(1:end-954+i);
 myewma1_new = ewma(sp01_new, mylambda1, mean(spR1_new)) ;   
 sigmahat_new(i) = (1-mylambda1)*spR1_new(end).^2+ mylambda1*myewma1_new.sigma_hat_1d(end).^2;
 end

figure(70)
plot(spR_test_date_new(end-9:end),r2_new(end-9:end))
hold on
plot(spR_test_date_new(end-9:end),sigmahat_new)
title('EWMA1 forecast 10 step ahead on new sample')
xlabel('Date')
ylabel('r^2')
legend('Observed', 'Forecast')

% evaluation 
myrmsfe_ewma1_new = sqrt(mean(r2_new(end-9:end)-sigmahat_new).^2);


%ARCH(8) new sample
% ARCH (0,8)
myarch8 = garch(0,8);
myarch8_est_new = estimate(myarch8, spR_sample_new, 'E0', spR_sample(1:10) );
 
varARCH8_new = infer(myarch8_est_new, spR_sample_new);

%ACF arch8
figure(71)
subplot(2,1,1)
autocorr(varARCH8_new)
subplot(2,1,2)
plot(varARCH8_new)

rmse_ARCH8 =  sqrt(mean((r2_new(1:end-10) -  varARCH8_new).^2));

figure(72)
plot(spR_sample_date_new(1:end-10), spR_sample_new(1:end-10))
hold on
plot(spR_date_new(1:end-10), varARCH8_new)

FvarARCH8_new = forecast(myarch8_est_new, 10, 'Y0',spR_sample_new);

figure(73)
plot(spR_test_date_new(end-9:end), r2_new(end-9:end))
hold on
plot(spR_test_date_new(end-9:end), FvarARCH8_new)
title('Forecast ARCH(8) on new sample')
xlabel('Time')
ylabel('r^2')
legend('Proxy volaitlity', 'Forecast ARCH(8)')

%evaluation
rmsfe_ARCH8F_new =  sqrt(mean(r2_new(end-9:end) -  FvarARCH8_new).^2);

% GARCH (1,2) new sample
mygarch3 = garch(1,2);
mygarch3_est_new = estimate(mygarch3, spR_sample_new, 'E0', spR_sample_new(1:10) );
varGARCH3_new = infer(mygarch3_est_new, spR_sample_new);
rmse_GARCH3_new =  sqrt(mean((r2(1:end-954) - varGARCH3_new).^2));
disp(['RMSE: ', num2str(rmse_GARCH3_new)]);
%forecast garch(1,2)
F_garch3_new = forecast(mygarch3_est_new,10,'Y0',spR_sample_new);
figure(74)
plot(spR_test_date_new(end-9:end),r2_new(end-9:end))
hold on
plot(spR_test_date_new(end-9:end),F_garch3_new)
title('Forecast GARCH(1,2) on new sample')
legend('Observed', 'Forecast')
ylabel('r^2')
xlabel('Date')

my_rmse_Fgarch3_new = sqrt(mean((r2(end-9:end)-F_garch3_new).^2));


save workspaceBLANDO