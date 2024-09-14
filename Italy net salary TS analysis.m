%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 341);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["ANNO", "TRIM", "REG", "CODPRO", "WAVQUA", "GRACOM", "SG4", "SG11", "SG13", "SG16", "SG18B", "SG18D", "SG18E", "SG18F", "SG18G", "SG24A", "SG24B", "SG25", "SG26", "SG27", "SG27A", "B1", "B2", "B3", "B3bis", "B4", "B4A", "B4bis", "B6", "B7", "B8", "B9", "B10", "B11", "C1", "C1bis", "C1A", "C1B", "C1D", "C4", "C5", "C6", "C7", "C9", "C10", "C14", "C16_c", "C18", "C19", "C20", "C21", "C22", "C23A", "C24", "C24bis", "C24ter", "C25", "C27", "C27A", "C28", "C29", "C29B", "C31", "C31A", "C32", "C33", "C34", "C35", "C36", "C36A", "C37", "C38", "C39", "C39A", "C40", "C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C53", "C55", "C56", "C57", "C59", "C59AA", "C60", "C61", "C62", "C73", "C74", "C75", "C76", "C77", "C78", "C79", "C80", "C81A", "C82", "C83", "D1", "D2", "D3", "D4", "D5", "D5A", "D5bis", "D7", "D9_c", "E1", "E2", "E3", "E4", "E6", "E7", "E8", "E13_c", "E14", "E15", "F1", "F2", "F3", "F4", "F5", "F6_01", "F6_02", "F6_03", "F6_04", "F6_05", "F6_06", "F6_07", "F6_08", "F6_09", "F6_10", "F6_11", "F6_12", "F6_96", "F6A_01", "F6A_02", "F6A_03", "F6A_96", "F7", "F8", "F9", "F10", "F10A", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F24A_01", "F24A_02", "F24A_03", "F24A_96", "F25", "F26", "F27", "F28", "F29_01", "F29_02", "F29_03", "F29_04", "F29_97", "F30", "F31", "F32", "F33", "F34", "F35", "F36", "F37", "F38", "F39", "F40", "F41", "F43", "F44", "F45", "F46", "F47", "F48", "F49", "F50", "G1", "G2", "G3", "G4_10", "G4_11", "G4_12", "G4_01", "G4_02", "G4_03", "G4_04", "G4_05", "G4_06", "G4_13", "G4_07", "G4_08", "G4_09", "G4_96", "G4_97", "G5", "G6", "G7_01", "G7_02", "G7_03", "G7_97", "G8_01", "G8_02", "G8_03", "G8_97", "G9", "H1", "H1B", "H1D", "H2A", "H2B", "H3", "H4", "H4A", "H5", "H6", "H7", "H8", "H9", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "I1", "I5", "I6", "I7", "I8", "I9", "I11_c", "I12", "I13", "RIP5", "RIP3", "ETAM", "CLETAD", "CLETAQ", "CLETAS", "CITTAD", "NASSES", "CITSES", "LLASES", "RAPSES", "COND3", "COND10", "DIPAUT", "DIPIND", "ASSOCC", "POSPRO", "DETIND", "PIEPAR", "DURATT", "INIATT", "ORELAV", "PROF1", "PROF3", "ISCO3D", "PROFM", "ATE2D", "CAT12", "CAT5", "CAT3", "LAVSPE", "REGSPE", "PROSPE", "TRACOM", "REGTRA", "PROTRA", "RETRIC", "INCDEC", "DIPAUS", "DIPINS", "PROF1S", "PROF3S", "PROFSM", "ATE2DS", "CAT12S", "CAT5S", "CAT3S", "DIPAUE", "DIPINE", "PROF1E", "PROF3E", "PROFEM", "ATE2DE", "CAT12E", "CAT5E", "CAT3E", "DIPAUA", "DIPINA", "DIPAUP", "DIPINP", "ATE2DP", "CAT12P", "CAT5P", "CAT3P", "DOVRIS", "REGPRE", "PROPRE", "ESPLAV", "DURNOC", "DURRIC", "DURAD", "HATLEV", "BLANK", "STACIM", "RELPAR", "AMATRI", "MFRFAM", "MFRIND", "NN2", "RPN2", "TFM", "TN2", "TISTUD", "EDULEV", "COISTR", "EDUCST", "COURAT", "COEFMI"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "categorical", "double", "double", "double", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "double", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "double", "double", "double", "double", "categorical", "double", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "double", "categorical", "categorical", "double", "double", "categorical", "double", "double", "categorical", "categorical", "categorical", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "categorical", "double", "double", "double", "double", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "double", "double", "double", "categorical", "categorical", "categorical", "categorical", "double", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "double", "double", "categorical", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "categorical", "categorical", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "categorical", "double", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "double", "double", "double", "double", "double", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["GRACOM", "SG18B", "SG18D", "SG18E", "SG18F", "SG18G", "SG24A", "SG24B", "SG27", "SG27A", "B2", "B3", "B3bis", "B4", "B4A", "B4bis", "B6", "B7", "B8", "B9", "B10", "B11", "C1bis", "C1A", "C1B", "C1D", "C4", "C5", "C6", "C7", "C19", "C21", "C22", "C23A", "C24bis", "C24ter", "C25", "C27A", "C28", "C29", "C29B", "C31A", "C33", "C34", "C36A", "C39", "C39A", "C40", "C41", "C56", "C57", "C62", "D2", "D3", "D4", "D5", "D5A", "D5bis", "D7", "D9_c", "E1", "E2", "E3", "E4", "E6", "E7", "E8", "E13_c", "E14", "E15", "F2", "F3", "F4", "F5", "F6_01", "F6_02", "F6_03", "F6_04", "F6_05", "F6_06", "F6_07", "F6_08", "F6_09", "F6_10", "F6_11", "F6_12", "F6_96", "F6A_01", "F6A_02", "F6A_03", "F6A_96", "F7", "F8", "F9", "F10", "F10A", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F24A_01", "F24A_02", "F24A_03", "F24A_96", "F25", "F26", "F27", "F28", "F29_01", "F29_02", "F29_03", "F29_04", "F29_97", "F30", "F31", "F32", "F33", "F34", "F35", "F36", "F37", "F38", "F39", "F40", "F41", "F43", "F44", "F45", "F46", "F47", "F48", "F49", "F50", "G2", "G3", "G4_10", "G4_11", "G4_12", "G4_01", "G4_02", "G4_03", "G4_04", "G4_05", "G4_06", "G4_13", "G4_07", "G4_08", "G4_09", "G4_96", "G4_97", "G5", "G6", "G8_01", "G8_02", "G8_03", "G8_97", "H1B", "H1D", "H2A", "H2B", "H3", "H4A", "H5", "H6", "H7", "H8", "H9", "H14", "H15", "H16", "H17", "H18", "H19", "I7", "I8", "NASSES", "CITSES", "LLASES", "RAPSES", "REGTRA", "PROTRA", "DIPAUS", "DIPINS", "PROF1S", "PROF3S", "PROFSM", "ATE2DS", "CAT12S", "CAT5S", "CAT3S", "DIPAUE", "DIPINE", "PROF1E", "PROF3E", "PROFEM", "ATE2DE", "CAT12E", "CAT5E", "CAT3E", "DIPAUA", "DIPINA", "DOVRIS", "REGPRE", "PROPRE", "ESPLAV", "DURNOC", "DURRIC", "DURAD", "BLANK", "COISTR"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["REGSPE", "PROSPE"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["REGSPE", "PROSPE"], "ThousandsSeparator", ",");

% Import the data
lfs20230222125415 = readtable("C:\Users\stepb\Desktop\METODI QUANTITATIVI PER L'ECONOMIA\PROBLEM SET\lfs-2023-02-22-12-54-15.csv", opts);

%% DATA SPECIFICATION

% denominazione dataset
data = lfs20230222125415;

% denominazione variabili 
salary = data.RETRIC;
education = categorical(data.TISTUD);
geography = categorical(data.REG);
geography_byzone = categorical(data.RIP5);
age = data.ETAM;
age_by5yclasses = categorical(data.CLETAQ);
gender = data.SG11;
workhours = data.ORELAV;

%% Q1  %%
% Statistiche descrittive SALARY
mean_salary = mean(salary); % media
median_salary = median (salary); % mediana
std_salary = std(salary); % deviazione standard
min_salary = min(salary); % valore minimo
max_salary = max(salary); % valore massimo
percentile05_salary = prctile(salary, 05); %primo 5%
percentile25_salary = prctile(salary, 25); % primo quartile
percentile50_salary = prctile(salary, 50); % mediana
percentile75_salary = prctile(salary,75); %terzo quartile
percentile95_salary = prctile(salary, 95); %ultimo 5%

disp(['Media del salario netto: ', num2str(mean_salary)]);
disp(['Mediana del salario netto: ', num2str(median_salary)]);
disp(['Deviazione standard del salario netto: ', num2str(std_salary)]);
disp(['Minimo salario netto: ', num2str(min_salary)]);
disp(['Massimo salario netto: ', num2str(max_salary)]);
disp(['Percentle 05 del salario netto: ', num2str(percentile05_salary)]);
disp(['Primo quartile del salario netto: ', num2str(percentile25_salary)]);
disp(['Secondo quartile salario netto: ', num2str(percentile50_salary)]);
disp(['Terzo quartile del salario netto: ', num2str(percentile75_salary)]);
disp(['Percentile 95 del salario netto: ', num2str(percentile95_salary)]);


% Statistiche descrittive SALARY - EDUCATION
stats_salary_edu = grpstats(data,{'TISTUD'},{'mean','median','std','min','max'},'DataVars', 'RETRIC');
disp(stats_salary_edu)

% Statistiche descrittive SALARY - GEOGRAPHY
stats_salary_geo = grpstats(data,{'REG'},{'mean','median','std','min','max'},'DataVars','RETRIC');
disp(stats_salary_geo)

% statistiche descrittive SALARY-EDUCATION+GEOGRAPHY
stats_salary_edugeo = grpstats(data,{'TISTUD', 'REG'},{'mean','median','std','min','max'},'DataVars','RETRIC');
disp(stats_salary_edugeo)


% Correlazione SALARY-EDUCATION e SALARY-GEOGRAPHY
corr_salaryedu = corr(data.RETRIC,data.TISTUD,'type','Spearman');
corr_salarygeo = corr(data.RETRIC,data.REG,'type','Spearman');

disp(['Correlazione tra salario e titolo di studio: ', num2str(corr_salaryedu )]);
disp(['Correlazione tra salario e regione d''appartenenza: ', num2str(corr_salarygeo )]);

% FREQUENZE 
N = size(data,1);

% FREQUENZE SALARIO 

[~,~,c]= unique(salary);
f_assolute_salary = accumarray(c,1);
f_relative_salary = f_assolute_salary/N;

figure (1)
subplot(1,2,1);
bar(f_assolute_salary, 'FaceColor', [0.2 0.5 0.7]);
title('Frequenze assolute');
subplot(1,2,2);
bar(f_relative_salary,  'FaceColor', [0.1 0.8 0.5]);
title('Frequenze relative');

% FREQUENZE TITOLO DI STUDIO

[~,~,c]= unique(education);
f_assolute_education = accumarray(c,1);
f_relative_education = f_assolute_education/N;

figure(2)
subplot(1,2,1);
bar(f_assolute_education);
xticklabels({'Nessun titolo', 'Licenza elementare', 'Licenza media', 'Diploma professionale di 2-3 anni', 'Diploma di maturità', 'Diploma di Accademia', 'Diploma universitario di due/tre anni','Laurea di primo livello (triennale)','Laurea specialistica/magistrale biennale','Laurea a ciclo unico'});
xlabel('Titolo di studio');
xtickangle(45);
ylabel('frequenza');
title('Frequenze assolute per titolo di studio');

subplot(1,2,2)
bar(f_relative_education);
xticklabels({'Nessun titolo', 'Licenza elementare', 'Licenza media', 'Diploma professionale di 2-3 anni', 'Diploma di maturità', 'Diploma di Accademia', 'Diploma universitario di due/tre anni','Laurea di primo livello (triennale)','Laurea specialistica/magistrale biennale','Laurea a ciclo unico'});
xlabel('Titolo di studio');
xtickangle(45);
ylabel('frequenza');
title('Frequenze relative per titolo di studio');


% FREQUENZE REGIONI 

[a,~,c]= unique(geography);
f_assolute_geography = accumarray(c,1);
f_relative_geography = f_assolute_geography/N;

figure(3)
subplot(1,2,1);
bar(f_assolute_geography);

xticks(1:length(unique(geography)));
xticklabels ({ 'Piemonte', 'Valle d'' Aosta', ' Lombardia', 'Trentino Alto Adige', 'Veneto', ' Friuli Venezia Giulia', 'Liguria','Emilia/Romagna','Toscana','Umbria', 'Marche', 'Lazio','Abruzzo','Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna' });
xlabel('Regione di appartenenza');
xtickangle(45);
ylabel('frequenza');
title('Frequenze assolute per regione');

subplot(1,2,2)
bar(f_relative_geography);
xticks(1:length(unique(geography)));
xticklabels(unique(geography));
xticklabels ({ 'Piemonte', 'Valle d'' Aosta', ' Lombardia', 'Trentino Alto Adige', 'Veneto', ' Friuli Venezia Giulia', 'Liguria','Emilia/Romagna','Toscana','Umbria', 'Marche', 'Lazio','Abruzzo','Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna' });
xlabel('Regione di appartenenza');
xtickangle(45);
ylabel('frequenza');
title('Frequenze relative per regione');


% GRAFICI Q1 

% ISTOGRAMMA E DENSITà SALARIO CONGIUNTI

figure(4)
histogram(salary, 'Normalization', 'pdf');

xlabel('Salario netto (euro)');
xtickangle(45);
ylabel('Frequenza');
title('Salario netto')
xlabel('Salario netto (euro)');
hold on
ksdensity(salary);

xtickangle(45);
ylabel('Frequenza');
title('Salario netto');

% GRAFICO A BARRE SALARIO DIVISO PER REGIONE E PER TITOLO DI STUDIO 

figure (5)
mean_salary = splitapply(@mean, salary, (data.REG));
b = bar(mean_salary);
b.FaceColor = 'flat';
b.CData(1,:) = [0 1 0.3];
b.CData(2,:) = [0 1 0.3];
b.CData(3,:) = [0 1 0.3];
b.CData(4,:) = [0 1 0.3];
b.CData(5,:) = [1 1 0.3];
b.CData(6,:) = [0 1 0.3];
b.CData(7,:) = [0 1 0.3];
b.CData(8,:) = [0 1 0.3];
b.CData(9,:) = [1 1 0.3];
b.CData(10,:) = [1. 0.6 0];
b.CData(11,:) = [1. 0.6 0];
b.CData(12,:) = [1 1 0.3];
b.CData(13,:) = [1 0.8 0.3];
b.CData(14,:) = [1 0.5 0];
b.CData(15,:) = [1 0.6 0];
b.CData(16,:) = [1 0.5 0];
b.CData(17,:) = [1 0.8 0.3];
b.CData(18,:) = [1 0.3 0];
b.CData(19,:) = [1 0.1 0];
b.CData(20,:) = [1 0.2 0];
xticks(1:length(unique(data.REG)));
xticklabels(unique(data.REG));
xticklabels ({ 'Piemonte', 'Valle d'' Aosta', ' Lombardia', 'Trentino Alto Adige', 'Veneto', ' Friuli Venezia Giulia', 'Liguria','Emilia-Romagna', 'Toscana', 'Umbria', 'Marche', 'Lazio', 'Abruzzo', 'Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna' });
xtickangle(45);
xlabel('Regione');
ylabel('Salario netto medio (euro)');
title('Salario netto medio per regione geografica');
annotation(figure(1),'line',[0.13 0.905], [0.873064837905237 0.871571072319202],'Color',[0.0745098039215686 0.623529411764706 1], 'LineWidth',1.4);
annotation(figure(1),'textbox',...
    [0.00992748538011696 0.7456608478802993 0.0907397660818713 0.0511221945137157],...
    'Color',[0.0745098039215686 0.623529411764706 1],...
    'String','MEDIA SALARIO',...
    'FontWeight','bold',...
    'FitBoxToText','off');

figure (6)
edu_mean = splitapply(@mean, salary, (data.TISTUD));
b = bar(edu_mean);
b.FaceColor = 'flat';
b.CData(5,:) = [1 1 0.3];
b.CData(6,:) = [0 1 0.3];
b.CData(7,:) = [0 1 0.3];
b.CData(8,:) = [1 1 0.3];
b.CData(9,:) = [0 1 0.3];
b.CData(10,:) = [0 1 0.3];
b.CData(1,:) = [1 0.2 0];
b.CData(2,:) = [1 0.3 0];
b.CData(3,:) = [1 0.4 0];
b.CData(4,:) = [1 0.5 0];
xticklabels({'Nessun titolo', 'Licenza elementare', 'Licenza media', 'Diploma professionale di 2-3 anni', 'Diploma di maturità', 'Diploma di Accademia', 'Diploma universitario di due/tre anni','Laurea di primo livello (triennale)','Laurea specialistica/magistrale biennale','Laurea a ciclo unico'});
xlabel('Titolo di studio');
xtickangle(45);
ylabel('Salario netto medio (euro)');
title('Salario netto medio per titolo di studio');
annotation(figure(2),'line',[0.13 0.905], [0.742064837905237 0.742571072319202],'Color',[0.0745098039215686 0.623529411764706 1], 'LineWidth',1.4);
annotation(figure(2),'textbox',...
    [0.00992748538011696 0.7456608478802993 0.0907397660818713 0.0511221945137157],...
    'Color',[0.0745098039215686 0.623529411764706 1],...
    'String','MEDIA SALARIO',...
    'FontWeight','bold',...
    'FitBoxToText','off');


% BOXPLOT SALARIO NETTO DIVISO PER AREA GEOGRAFICA

figure (7)
boxplot(salary, geography);
title('Box plot del salario netto per area geografica');
xticklabels ({ 'Piemonte', 'Valle d'' Aosta', ' Lombardia', 'Trentino Alto Adige', 'Veneto', ' Friuli Venezia Giulia', 'Liguria','Emilia-Romagna', 'Toscana', 'Umbria', 'Marche', 'Lazio', 'Abruzzo', 'Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna' });
xlabel('Regione di appartenenza');
xtickangle(45);
ylabel('Salario netto');

% BOXPLOT SALARIO NETTO PER TITOLO DI STUDIO 

figure (8)
boxplot(salary, education);
title('Box plot del salario netto per titolo di studio');
xticklabels({'Nessun titolo', 'Licenza elementare', 'Licenza media', 'Diploma professionale di 2-3 anni', 'Diploma di maturità', 'Diploma di Accademia', 'Diploma universitario di due/tre anni','Laurea di primo livello (triennale)','Laurea specialistica/magistrale biennale','Laurea a ciclo unico'});
xlabel('Titolo di studio');
xtickangle(45);
ylabel('Salario netto');

figure (9)
histogram(gender);
xticks([1, 2]);
xticklabels({'Uomo', 'Donna'});
xlabel('Genere');
ylabel('Frequenza');
title('Distribuzione del genere nel campione');

% TEST NORMALITà DISTRIBUZIONE 

[h, p, kslstat] = lillietest(salary);
disp(['Lilliefors-test per la normalità della distribuzione dei salari: h = ', num2str(h), ', p = ', num2str(p), ', kslstat = ', num2str(kslstat)])

% il test restituisce un p-value molto basso, indicando che la distribuzione dei salari netti non segue una distribuzione normale

%% Q2  DIFFERENZA STATISTIAMENTE SIGNIFICATIVA PER SOTTOGRUPPI di EDUCATION

% ANOVA per TISTUD
tistud_groups = splitapply(@mean, salary, data.TISTUD);
[p_tistud, tbl_tistud, stats_tistud] = anova1(salary, education);

% ANOVA per REG
reg_groups = splitapply(@mean, salary, data.REG);
[p_reg, tbl_reg, stats_reg] = anova1(salary, geography);

% Creare due sottogruppi per EDUCATION

groups = table(salary, education);
[~,~,idx] = unique(groups.education);
mean_salary = splitapply(@mean, groups.salary, idx);
subset_1edu = groups(idx <= 4,:);
subset_2edu = groups(idx >= 5,:);

mean_subset_1edu = mean(subset_1edu.salary);
mean_subset_2edu = mean(subset_2edu.salary);

% Calcolare le deviazioni standard dei sottogruppi
std_subset_1geo = std(subset_1edu.salary);
std_subset_2geo = std(subset_2edu.salary);

% Calcolare il test t per le due medie
[h,p,ci,stats] = ttest2(subset_1edu.salary,subset_2edu.salary);

% Visualizzazione del risultato del test
if p < 0.05
    disp('Le medie sono significativamente diverse')
else
    disp('Le medie non sono significativamente diverse')
end


% RISULTATI SIGNIFICATIVAMENTE DIVERSE PER LE MEDIE DEL SALARIO DI
% LAUREATI E NON LAUREATI 


%% Q3 modalità titolo di studio del 5 e 95 percentile del salario









%% Q4 REGRESSIONE BIVARIATA SALARIO-ETà 

lm1 = fitlm(age, salary,'linear')

S = corr(age,salary);
var_cov = cov(age,salary);
b1 = var_cov(1,2)/ var(age);
b0 = mean(salary)- b1*mean(age);

y_hat = b0 + b1*age;

residui = salary -y_hat;
ksdensity(residui)
mean(residui)


% residui -> ^2 -> mean(residui.^2) -> sqrt(mean(residui.^2))
RMSE = sqrt(mean(residui.^2));

% residui -> ^2 -> sum(residui.^2) -> (1/(n-2))* sum(residui.^2) -> sqrt((1/(n-2))* sum(residui.^2)))
n = length(residui);
k = 1;
SER = sqrt((1/(n-k-1))* sum(residui.^2));

% R2 -> cor(X,y)^2
myR2 = (corr(age,salary))^2;

% R2 -> num: explain, den : tot
% num: sum((y_hat - mean(y_hat)).^2)
%den: sum((y - mean(y)).^2)

myR2_bis = (sum((y_hat - mean(y_hat)).^2))/(sum((salary- mean(salary)).^2));

%MATRICE DI CORRELAZIONE PER VARIABILI QUANTITATIVE
numeric_vars = ["ETAM", "ORELAV", "DURATT", "RETRIC"];
corrplot(data{:,numeric_vars},'testR','on','type','Spearman');

numeric_vars2 = ["ETAM", "TISTUD"];
corrplot(data{:,numeric_vars2},'testR','on','type','Spearman');


%% Q5 REGRESSIONE TRIVARIATA 
corr_ETAEDU = corr(data.ETAM,data.TISTUD,'type','Spearman')

lm2= fitlm(data, ['RETRIC ~ ETAM+TISTUD'])
plot(lm2)




%% Q6 REGRESSIONE MULTIVARIATA 

lm3= fitlm(data, ['RETRIC ~ ETAM+REG+SG11+CITTAD+DIPIND+LAVSPE+POSPRO+DETIND+PIEPAR+ORELAV'])
plot(lm3)




