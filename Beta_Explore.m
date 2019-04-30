% T. Alongi 2019-03-27 Look at FOIA request Beta Prod spreadsheet
clc; clear; clf; close all;

tbl = readtable('/auto/home/talongi/Pvf/FOIA_request/FOIA_UCSC_Beta Prod_cp.xlsx');
cat = readtable('/auto/home/talongi/Pvf/Data_tables/Eq_cat/SCSN_alt.txt');
eureka_loc = [33.56380336	-118.1173923];

% FOIA Table
    % assign columns
    date = tbl.PRODUCTION_DATE;
    well_number = str2double(tbl.API_WELL_NUMBER); 
    platform = tbl.COMPLEX_ID_NUM; %51009 = Ellen; 51015 = Eureka
        ellen = (platform == 51009);
        eureka = (platform == 51015);

    %production
    oil_prod = tbl.MON_O_PROD_VOL;
    gas_prod = tbl.MON_G_PROD_VOL;
    wtr_prod = tbl.MON_WTR_PROD_VOL;

    %injection
    inj_vol = tbl.INJECTION_VOLUME;
    inj_type = tbl.WELL_STAT_CD; %04 = water injection; 05 = water disposal
    
% Catalog table
    yr = cat.Var1;
    mon = cat.Var2;
    day = cat.Var3;
    hr = cat.Var4;
    min = cat.Var5;
    sec = cat.Var6;

    lon = cat.Var9;
    lat = cat.Var8;
    depth = cat.Var10;
    M = cat.Var11;
  
    datematrix = [yr, mon, day, hr, min, sec];
    date_cat = datetime(datematrix);
    

%%
uniq_date = unique(date);
uniq_well_number = unique(well_number);

% two year window
% min_date = datetime('1-Nov-1998'); 
% max_date = datetime('1-Dec-2000');

min_date = datetime('1-Aug-2010'); 
max_date = datetime('1-Oct-2011');

%FOIA data
yr_win = min_date < date & date < max_date; 

% EQ data
    % coord limits
            N_lat = 34.0;
    E_lon = -118.6; W_lon = -117.8;
            S_lat = 33.0;

    coord_lim = lat < N_lat...
        & lat > S_lat...
        & lon > E_lon...
        & lon < W_lon;

    % time window
    eq_time_window = min_date < date_cat...
        & date_cat < max_date; 
    
    % compute arclen on Earth sphere in Km
    [arclen, az] = distance(eureka_loc(1), eureka_loc(2), ...
        lat, lon, ...
        referenceSphere('Earth','kilometer'));
    
    % radial distance window
    dist_tol = 30; % km from eureka rig
    dist_window = (arclen <= dist_tol);
    
    %combine all masks 
%     eq_mask = eq_time_window & dist_window;
    eq_mask = dist_window;

        sprintf('number of events = %d',sum(eq_mask))

%%
        
% make a matrix of monthly sums for all wells associated with eureka
time_steps = uniq_date(uniq_date < '1-Dec-2000');
cum_matrix = zeros(length(time_steps), 5);

for i = 1:length(time_steps)
    indexes = (date == uniq_date(i));
    
    cum_oil = sum(oil_prod(indexes & eureka));
    cum_gas = sum(gas_prod(indexes & eureka));
    cum_wtr = sum(wtr_prod(indexes & eureka));
    cum_inj = sum(inj_vol(indexes & eureka));
    
    cum_matrix(i,1) = datenum(uniq_date(i)); %convert to datenum for matrix
                                             %convert back with datestr
    cum_matrix(i,2) = cum_oil;
    cum_matrix(i,3) = cum_gas;
    cum_matrix(i,4) = cum_wtr;
    cum_matrix(i,5) = cum_inj;
    
end


[n, edges] = histcounts(date_cat(eq_mask), 'BinMethod', 'month');
disp(edges(1))

% time per Eureka shut down
shutdown = [datetime('5-Jun-1999') datetime('1-Apr-2008')];
shutdown_mask = edges > shutdown(1) ...
    & edges < shutdown(2);

% plots
figure(84)
plt_base = 0;
plt_height = 12;

    plot(edges(1:end-1), n , 'k:'); hold on
    plot(edges(1:end-1), smooth( n, 5), 'b')
    plot(edges(shutdown_mask), smooth(n(shutdown_mask), 5), 'r')

        title(sprintf('Earthquake Rate with in %i km of Eureka Platform \n Total Number of Events %i' ,...
            dist_tol, sum(eq_mask)))

        xlabel('Time'); ylabel('Earthquakes / Month');
        legend('Raw Data', 'Smoothed', 'Eureka Shutdown')

        ylim([plt_base, plt_height])

hold off

%% stem plot of time,magnitude
figure(85)
    stem(date_cat(eq_mask), M(eq_mask), 'b'); hold on
%     plot(date_cat(eq_mask), smooth(M(eq_mask)), 'r-', 'LineWidth', 1.6);
    bar(datetime('5-Jun-1999'), max(M(eq_mask)),'r')

    T = 'Eureka Shutdown'
    text(datetime('20-Jun-1999'), max(M(eq_mask)), T)
    ylabel('Magnitude')
    hold off

%%
figure(86)    
clf

plot(edges(1:end-1), n, 'ko-', 'LineWidth', 2.2); hold on
    plot(edges(1:end-1), smooth(n), 'k:', 'LineWidth', 1.8)
addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,2), 'LineWidth',1.8)
addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,3), 'LineWidth',1.8)
addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,4), 'LineWidth',1.8)
addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,5), 'LineWidth',1.8)


xlim([(edges(1)), (edges(end))]);

title(sprintf('Eureka Production, Injection, & Seismicity Within %i km \n Number of Events = %d' ,...
    dist_tol, sum(eq_mask)))

lbl = {'Number of Events/Month','Smoothed Events/Month', 'Oil Production', 'Gas Production', 'Water Production', 'Injection'}

addaxislabel(1, lbl(1))
addaxislabel(2, lbl(2))
addaxislabel(3, lbl(3))
addaxislabel(4, lbl(4))
addaxislabel(5, lbl(5))


legend(lbl{1}, lbl{2}, lbl{3}, lbl{4}, lbl{5}, lbl{6})
hold off


%% Plot of Production & injection

figure(1)v 
%Ellen
    subplot(4,2,1)
        % time window
        stem(date(yr_win & ellen), oil_prod(yr_win & ellen), 'k')
        title({'Ellen', 'Oil Production'})

    subplot(4,2,3)
        stem(date(yr_win&ellen), gas_prod(yr_win&ellen), 'g');
        title('Gas Production')

    subplot(4,2,5)
        stem(date(yr_win&ellen), wtr_prod(yr_win&ellen), 'b');
        title('Water Production')

    subplot(4,2,7)
        stem(date(yr_win & ellen), inj_vol(yr_win & ellen), 'r');
        title('Injection Volume')

%Eureka
    subplot(4,2,2)
        stem(date(yr_win &eureka), oil_prod(yr_win & eureka),'k');
        title({'Eureka','Oil Production'})

    subplot(4,2,4)
        stem(date(yr_win & eureka), gas_prod(yr_win &eureka), 'g');
        title('Gas Production')

    subplot(4,2,6)
        stem(date(yr_win&eureka), wtr_prod(yr_win&eureka), 'b');
        title('Water Production')

    subplot(4,2,8)
        stem(date(yr_win & eureka), inj_vol(yr_win & eureka), 'r');
        title('Injeciton Volume')
hold off


%% Albacore Deployment
uniq_date = unique(date);
uniq_well_number = unique(well_number);

%
min_date = datetime('1-Jun-2010'); max_date = datetime('1-Nov-2011');
yr_win = min_date < date & date < max_date; % Albacore deployment

% EQ data
    % coord limits
            N_lat = 34.0;
    E_lon = -118.6; W_lon = -117.8;
            S_lat = 33.0;
    
    % coord window
    coord_lim = lat < N_lat...
        & lat > S_lat...
        & lon > E_lon...
        & lon < W_lon;

    % time window
    eq_time_window = min_date < date_cat...
        & date_cat < max_date; 
    
    % compute arclen on Earth sphere in Km
    [arclen, az] = distance(eureka_loc(1), eureka_loc(2), ...
        lat, lon, ...
        referenceSphere('Earth','kilometer'));
    
    % radial distance window
    dist_tol = 30; % km from eureka rig
    dist_window = (arclen <= dist_tol);
    
    %combine all masks 
    eq_mask = coord_lim & eq_time_window & dist_window;
        sprintf('number of events = %d',sum(eq_mask))


time_steps = uniq_date(uniq_date > '1-Dec-2000')
cum_matrix = zeros(length(time_steps), 5);

for i = 1:length(time_steps)
    indexes = (date == time_steps(i));
    
    cum_oil = sum(oil_prod(indexes & eureka));
    cum_gas = sum(gas_prod(indexes & eureka));
    cum_wtr = sum(wtr_prod(indexes & eureka));
    cum_inj = sum(inj_vol(indexes & eureka));
    
    cum_matrix(i,1) = datenum(time_steps(i)); %convert to datenum for matrix
                                             %convert back with datestr
    cum_matrix(i,2) = cum_oil;
    cum_matrix(i,3) = cum_gas;
    cum_matrix(i,4) = cum_wtr;
    cum_matrix(i,5) = cum_inj;
    
end


[n, edges] = histcounts(date_cat(eq_mask), 'BinMethod', 'month');
disp(edges(1))


figure(87)    
clf
    plot(edges(2:end), n, 'ko-', 'LineWidth', 2.2)
    addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,2), 'LineWidth',1.8)
    addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,3), 'LineWidth',1.8)
    addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,4), 'LineWidth',1.8)
    addaxis(datetime(datestr(cum_matrix(:,1))), cum_matrix(:,5), 'LineWidth',1.8)


        xlim([(edges(1)), (edges(end))]);

        title(sprintf('Eureka Production, Injection, & Seismicity Within %i km \n Number of Events = %d' ,...
            dist_tol, sum(eq_mask)))

        lbl = {'Number of Events/Month', 'Oil Production', 'Gas Production', 'Water Production', 'Injection'};

        addaxislabel(1, lbl(1))
        addaxislabel(2, lbl(2))
        addaxislabel(3, lbl(3))
        addaxislabel(4, lbl(4))
        addaxislabel(5, lbl(5))

        legend(lbl{1}, lbl{2}, lbl{3}, lbl{4}, lbl{5})
hold off

%% Make Tables of events before leak, during shutdown, after repair

before_mask = date_cat < shutdown(1) & eq_mask; sum(before_mask)
during_mask = date_cat > shutdown(1) & date_cat < shutdown(2) & eq_mask; sum(during_mask)
after_mask = date_cat > shutdown(2) & eq_mask; sum(after_mask)

% make new tables applying masks
cat_b4 = cat(before_mask,:);
cat_dur = cat(during_mask,:);
cat_aft = cat(after_mask,:);

writetable(cat_b4, 'catalog_before_shutdown.txt', 'Delimiter', ' ')
writetable(cat_dur, 'catalog_during_shutdown.txt', 'Delimiter', ' ')
writetable(cat_aft, 'catalog_after_shutdown.txt', 'Delimiter', ' ')

%% make catalog of all events within tolerance distance of eureka

tol_cat = cat(eq_mask,:);

evnt_type = tol_cat.Var24; %le = local, qb = quarry, re = regional
loc_meth = str(tol_cat.Var25); % ct = xcorr, 3d = 3d velocity model, xx= not relocated
n_PnS = tol_cat.Var12;
near_stn = tol_cat.Var13;
rms = tol_cat.Var14;

hrz_error = tol_cat.Var20;
dep_error = tol_cat.Var21;


figure(99)
    subplot(4,1,1)
    histogram(hrz_error(hrz_error < 98), 'Normalization', 'pdf')
    title(sprintf('Avg Horizontal Error %2.1f +/- %0.1f km', ...
        mean(hrz_error(hrz_error < 98)), std(hrz_error(hrz_error < 98))))

    subplot(4,1,2)
    histogram(dep_error(dep_error < 98), 'Normalization', 'pdf')
    title(sprintf('Avg Depth Error %2.1f +/- %0.1f km', ...
        mean(dep_error(dep_error < 98)), std(dep_error(dep_error < 98))))

    sublpot(4,1,3)
    histogram
    
    subplot(4,1,4)
    histogram(near_stn(near_stn < 100))





