log_prefix_path = '../py/'; % relative path to log_prefix directory (IF NOT EMPTY, MUST HAVE TRAILING SLASH /
log_prefix = 'classical_vs_classical'; % directory containing mc log files

%% INTERPRET MONTE CARLO DATA

configfile = strcat(log_prefix_path, log_prefix, '/configuration.txt');
fileID = fopen(configfile,'r');
formatSpec = '%f';
config = fscanf(fileID, formatSpec);
fclose(fileID);

num_runs = round(config(1));

teamAscores = zeros(1, num_runs);
teamBscores = zeros(1, num_runs);
dmg_A1_toA2 = zeros(1, num_runs);
dmg_A1_toB1 = zeros(1, num_runs);
dmg_A1_toB2 = zeros(1, num_runs);
dmg_A2_toA1 = zeros(1, num_runs);
dmg_A2_toB1 = zeros(1, num_runs);
dmg_A2_toB2 = zeros(1, num_runs);
dmg_B1_toA1 = zeros(1, num_runs);
dmg_B1_toA2 = zeros(1, num_runs);
dmg_B1_toB2 = zeros(1, num_runs);
dmg_B2_toA1 = zeros(1, num_runs);
dmg_B2_toA2 = zeros(1, num_runs);
dmg_B2_toB1 = zeros(1, num_runs);

for i=1:1:num_runs
    
    logfile = strcat(log_prefix_path, log_prefix, '/mc_run_', num2str(i), '.log');
    logdata = read_log(logfile, 29);
    
    teamAscores(i) = logdata(2,end);
    teamBscores(i) = logdata(3,end);
    
    dmg_A1_toA2(i) = logdata(15,end);
    dmg_A1_toB1(i) = logdata(16,end);
    dmg_A1_toB2(i) = logdata(17,end);
    dmg_A2_toA1(i) = logdata(18,end);
    dmg_A2_toB1(i) = logdata(20,end);
    dmg_A2_toB2(i) = logdata(21,end);
    dmg_B1_toA1(i) = logdata(22,end);
    dmg_B1_toA2(i) = logdata(23,end);
    dmg_B1_toB2(i) = logdata(25,end);
    dmg_B2_toA1(i) = logdata(26,end);
    dmg_B2_toA2(i) = logdata(27,end);
    dmg_B2_toB1(i) = logdata(28,end);
    
end

figure(1)
subplot(1,2,1)
hist(teamAscores)
title('Team A Scores')
subplot(1,2,2)
hist(teamBscores)
title('Team B Scores')

figure(2)
subplot(4,3,1)
hist(dmg_A1_toA2);
title('Damage A1 Imposed on A2')
subplot(4,3,2)
hist(dmg_A1_toB1)
title('Damage A1 Imposed on B1')
subplot(4,3,3)
hist(dmg_A1_toB2)
title('Damage A1 Imposed on B2')
subplot(4,3,4)
hist(dmg_A2_toA1)
title('Damage A2 Imposed on A1')
subplot(4,3,5)
hist(dmg_A2_toB1)
title('Damage A2 Imposed on B1')
subplot(4,3,6)
hist(dmg_A2_toB2)
title('Damage A2 Imposed on B2')
subplot(4,3,7)
hist(dmg_B1_toA1)
title('Damage B1 Imposed on A1')
subplot(4,3,8)
hist(dmg_B1_toA2)
title('Damage B1 Imposed on A2')
subplot(4,3,9)
hist(dmg_B1_toB2)
title('Damage B1 Imposed on B2')
subplot(4,3,10)
hist(dmg_B2_toA1)
title('Damage B2 Imposed on A1')
subplot(4,3,11)
hist(dmg_B2_toA2)
title('Damage B2 Imposed on A2')
subplot(4,3,12)
hist(dmg_B2_toB1)
title('Damage B2 Imposed on B1')