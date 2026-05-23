%% chua_procedimiento_numerico.m
clear; clc; format long g

%% 1. Parametros del sistema
par.alpha = 8.4562;
par.beta  = 12.0732;
par.gamma = 0.0052;
par.m0    = -0.1768;
par.m1    = -1.1468;

tolJ = 1e-7;
Xeq  = zeros(3,1);

%% 2. Forma Lur'e obtenida numericamente desde el modelo
[P,q] = numeric_lure_matrices(@(X,ps) chua_model(X,ps,par),Xeq,tolJ);
r = [1 0 0];
I = eye(3);

W = @(p) r*((P - p*I)\q);

testX  = [0.37; -0.21; 0.58];
testPs = -0.42;
errLure = norm(P*testX + q*testPs - chua_model(testX,testPs,par));

%% 3. Raices de Im(W(i omega))=0 y ganancia k
omegaRoots = find_roots_positive(@(om) imag(W(1i*om)),1e-5,50,40000);
kRoots = zeros(size(omegaRoots));

for j = 1:numel(omegaRoots)
    kRoots(j) = -1/real(W(1i*omegaRoots(j)));
end

idx = 1;                       % rama de menor frecuencia
omega0 = omegaRoots(idx);
k      = kRoots(idx);

%% 4. Amplitud a0 por balance armonico numerico
Phi = @(A) harmonic_phi(A,k,par);
aRoots = find_roots_positive(Phi,1+1e-8,50,20000);
a0 = aRoots(1);

%% 5. Matriz modificada P0 y modo real d
P0 = P + k*(q*r);
eigP0 = eig(P0);

[~,idxReal] = min(abs(imag(eigP0)));
d = -real(eigP0(idxReal));

A = [0 -omega0 0;
     omega0 0 0;
     0 0 -d];

%% 6. Calculo numerico de h, b1, b2 por ajuste de transferencia
[h,b1,b2,errTF] = canonical_params(P0,q,r,A);

b = [b1; b2; 1];
c = [1; 0; -h];

%% 7. Matriz de transformacion S
[S,errS] = solve_S(P0,A,b,c,q,r);

Y0 = [a0; 0; 0];
X0 = S*Y0;

%% 8. Verificaciones numericas
errDyn = norm(P0*S - S*A,'fro');
errB   = norm(S*b - q);
errC   = norm(r*S - c.');

%% 9. Salida ordenada
disp('====================================================')
disp('RESULTADOS NUMERICOS DEL PROCEDIMIENTO')
disp('====================================================')

disp(' ')
disp('1) Matrices Lure obtenidas numericamente')
disp('P =')
disp(P)
disp('q =')
disp(q)
disp('r =')
disp(r)
fprintf('||P*X + q*psi - F|| = %.3e\n',errLure)

disp(' ')
disp('2) Raices positivas de Im(W(i omega))=0')
T = table(omegaRoots(:),kRoots(:), ...
    'VariableNames',{'omega','k'});
disp(T)

disp(' ')
disp('3) Rama seleccionada')
fprintf('omega0 = %.16f\n',omega0)
fprintf('k      = %.16f\n',k)

disp(' ')
disp('4) Amplitud armonica')
fprintf('a0     = %.16f\n',a0)
fprintf('Phi(a0)= %.3e\n',Phi(a0))

disp(' ')
disp('5) Matriz P0 y autovalores')
disp('P0 =')
disp(P0)
disp('eig(P0) =')
disp(eigP0)
fprintf('d      = %.16f\n',d)

disp(' ')
disp('6) Parametros canonicos')
fprintf('h      = %.16f\n',h)
fprintf('b1     = %.16f\n',b1)
fprintf('b2     = %.16f\n',b2)
fprintf('error transferencia = %.3e\n',errTF)

disp(' ')
disp('7) Matriz S')
disp('S =')
disp(S)

disp(' ')
disp('8) Condicion inicial obtenida desde X0 = S*[a0;0;0]')
disp('X0 =')
disp(X0)

disp(' ')
disp('9) Verificaciones finales')
fprintf('||P0*S - S*A||_F = %.3e\n',errDyn)
fprintf('||S*b - q||      = %.3e\n',errB)
fprintf('||r*S - c^T||    = %.3e\n',errC)
fprintf('error sistema S  = %.3e\n',errS)

%% 10. Graficas del procedimiento numerico
EXPORT_FIGS = true;
EXPORT_PDF  = false;
outDir = "chua_entero_outputs";

if ~exist(outDir,'dir')
    mkdir(outDir);
end

omegaPlot = linspace(1e-4, max(10,5*omega0), 5000);
Wplot = zeros(size(omegaPlot));

for j = 1:numel(omegaPlot)
    Wplot(j) = W(1i*omegaPlot(j));
end

PhiPlotA = linspace(1.0001, max(20,1.5*a0), 2500);
PhiPlot = arrayfun(Phi,PhiPlotA);

plot_transfer_parts(omegaPlot,Wplot,omegaRoots,omega0,k,outDir,EXPORT_FIGS,EXPORT_PDF);
plot_nyquist_integer(Wplot,omega0,W(1i*omega0),k,outDir,EXPORT_FIGS,EXPORT_PDF);
plot_phi_amplitude(PhiPlotA,PhiPlot,a0,Phi(a0),outDir,EXPORT_FIGS,EXPORT_PDF);
plot_eigs_P0(eigP0,omega0,d,outDir,EXPORT_FIGS,EXPORT_PDF);

Tper = 2*pi/omega0;
tHar = linspace(0,4*Tper,2000);
YHar = [
    a0*cos(omega0*tHar);
    a0*sin(omega0*tHar);
    zeros(size(tHar))
];

XHar = S*YHar;
plot_harmonic_seed(tHar,XHar,omega0,a0,outDir,EXPORT_FIGS,EXPORT_PDF);

%% 11. Simulacion entera: lineal modificado vs sistema original
tFinal = 80;
tspan = [0 tFinal];

sat = @(u) 0.5*(abs(u + 1) - abs(u - 1));
psi_pwl = @(sigma) (par.m0 - par.m1).*sat(sigma);

fLin = @(t,X) P0*X;
fOrg = @(t,X) P*X + q*psi_pwl(r*X);

opts = odeset('RelTol',1e-9,'AbsTol',1e-11);

[tLin,XLin] = ode45(fLin,tspan,X0,opts);
[tOrg,XOrg] = ode45(fOrg,tspan,X0,opts);

plot_integer_comparison(tLin,XLin,tOrg,XOrg,X0,outDir,EXPORT_FIGS,EXPORT_PDF);

fprintf('\nGraficas generadas en la carpeta:\n');
fprintf('  %s\n',outDir);

function F = chua_model(X,ps,par)
    x = X(1); y = X(2); z = X(3);
    F = [
        par.alpha*(y - x - par.m1*x - ps);
        x - y + z;
       -(par.beta*y + par.gamma*z)
    ];
end

function [P,q] = numeric_lure_matrices(Ffun,X0,epsJ)
    n = numel(X0);
    P = zeros(n,n);

    for j = 1:n
        ej = zeros(n,1);
        ej(j) = 1;
        Fp = Ffun(X0 + epsJ*ej,0);
        Fm = Ffun(X0 - epsJ*ej,0);
        P(:,j) = (Fp - Fm)/(2*epsJ);
    end

    qp = Ffun(X0, epsJ);
    qm = Ffun(X0,-epsJ);
    q  = (qp - qm)/(2*epsJ);
end

function rootsOut = find_roots_positive(fun,a,b,N)
    grid = linspace(a,b,N);
    vals = arrayfun(fun,grid);

    rootsOut = [];

    for j = 1:N-1
        if ~isfinite(vals(j)) || ~isfinite(vals(j+1))
            continue
        end

        if vals(j) == 0
            root = grid(j);
        elseif vals(j)*vals(j+1) < 0
            root = fzero(fun,[grid(j),grid(j+1)]);
        else
            continue
        end

        if root > a && root < b
            if isempty(rootsOut) || min(abs(rootsOut-root)) > 1e-7
                rootsOut(end+1,1) = root; %#ok<AGROW>
            end
        end
    end

    rootsOut = sort(rootsOut);
end

function val = harmonic_phi(A,k,par)
    sat = @(u) max(-1,min(1,u));

    f = @(th) ((par.m0-par.m1).*sat(A*cos(th)) ...
        - k.*A.*cos(th)).*cos(th);

    th0 = acos(1/A);
    pts = [0 th0 pi-th0 pi+th0 2*pi-th0 2*pi];
    pts = unique(max(0,min(2*pi,pts)));

    val = 0;
    for j = 1:numel(pts)-1
        val = val + integral(f,pts(j),pts(j+1), ...
            'AbsTol',1e-12,'RelTol',1e-12);
    end
end

function [h,b1,b2,errTF] = canonical_params(P0,q,r,A)
    I = eye(3);
    e1 = [1;0;0];
    e2 = [0;1;0];
    e3 = [0;0;1];

    W0 = @(p) r*((P0 - p*I)\q);

    sampleP = [0.3 0.7 1.1 1.7 2.3 3.1];
    M = [];
    y = [];

    for p = sampleP
        R = (A - p*I)\eye(3);

        col_h  = -e3.'*R*e3;
        col_b1 =  e1.'*R*e1;
        col_b2 =  e1.'*R*e2;

        row = [col_h col_b1 col_b2];
        rhs = W0(p);

        M = [M; real(row); imag(row)]; %#ok<AGROW>
        y = [y; real(rhs); imag(rhs)]; %#ok<AGROW>
    end

    u = M\y;

    h  = u(1);
    b1 = u(2);
    b2 = u(3);

    errTF = norm(M*u-y);
end

function [S,errS] = solve_S(P0,A,b,c,q,r)
    n = 3;
    idx = @(i,j) (i-1)*n + j;

    M = [];
    y = [];

    for i = 1:n
        for j = 1:n
            row = zeros(1,n*n);

            for ell = 1:n
                row(idx(ell,j)) = row(idx(ell,j)) + P0(i,ell);
                row(idx(i,ell)) = row(idx(i,ell)) - A(ell,j);
            end

            M = [M; row]; %#ok<AGROW>
            y = [y; 0]; %#ok<AGROW>
        end
    end

    for i = 1:n
        row = zeros(1,n*n);
        for ell = 1:n
            row(idx(i,ell)) = row(idx(i,ell)) + b(ell);
        end
        M = [M; row]; %#ok<AGROW>
        y = [y; q(i)]; %#ok<AGROW>
    end

    for j = 1:n
        row = zeros(1,n*n);
        for ell = 1:n
            row(idx(ell,j)) = row(idx(ell,j)) + r(ell);
        end
        M = [M; row]; %#ok<AGROW>
        y = [y; c(j)]; %#ok<AGROW>
    end

    svec = M\y;
    S = reshape(svec,[n,n]).';

    errS = norm(M*svec-y);
end

function plot_transfer_parts(omega,Wval,omegaRoots,omega0,k,outDir,doExport,doPDF)

    fig = figure('Color','w','Position',[100 100 1000 650]);
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    W0 = interp1(omega,Wval,omega0,'linear','extrap');

    nexttile;
    plot(omega,real(Wval),'LineWidth',1.2);
    hold on; grid on; box on;
    yline(-1/k,'--','LineWidth',1.0);
    plot(omega0,real(W0),'ro','MarkerSize',7,'LineWidth',1.5);

    for j = 1:numel(omegaRoots)
        xline(omegaRoots(j),':','LineWidth',0.8);
    end

    xlabel('\omega','Interpreter','tex');
    ylabel('Re\{W(i\omega)\}','Interpreter','tex');
    title('Parte real de la transferencia');
    legend('Re\{W\}','-1/k','\omega_0','Location','best');

    nexttile;
    plot(omega,imag(Wval),'LineWidth',1.2);
    hold on; grid on; box on;
    yline(0,'--','LineWidth',1.0);
    plot(omega0,imag(W0),'ro','MarkerSize',7,'LineWidth',1.5);

    for j = 1:numel(omegaRoots)
        xline(omegaRoots(j),':','LineWidth',0.8);
    end

    xlabel('\omega','Interpreter','tex');
    ylabel('Im\{W(i\omega)\}','Interpreter','tex');
    title('Parte imaginaria de la transferencia');
    legend('Im\{W\}','0','\omega_0','Location','best');

    if doExport
        exportgraphics(fig,fullfile(outDir,"01_transferencia_ReIm.png"),'Resolution',220);
        if doPDF
            exportgraphics(fig,fullfile(outDir,"01_transferencia_ReIm.pdf"),'ContentType','vector');
        end
    end
end

function plot_nyquist_integer(Wval,omega0,W0,k,outDir,doExport,doPDF)

    fig = figure('Color','w','Position',[100 100 850 650]);

    plot(real(Wval),imag(Wval),'LineWidth',1.2);
    hold on; grid on; box on;

    plot(real(W0),imag(W0),'ko','MarkerFaceColor','w','MarkerSize',8,'LineWidth',1.5);
    plot(-1/k,0,'rx','MarkerSize',11,'LineWidth',2);

    yline(0,'--','LineWidth',0.8);
    xline(0,'--','LineWidth',0.8);

    text(real(W0),imag(W0),sprintf('  \\omega_0=%.4f',omega0), ...
        'Interpreter','tex','FontSize',10);
    text(-1/k,0,'  -1/k','Interpreter','tex','FontSize',10);

    xlabel('Re\{W(i\omega)\}','Interpreter','tex');
    ylabel('Im\{W(i\omega)\}','Interpreter','tex');
    title('Nyquist numerico de W(i\omega)');
    legend('W(i\omega)','W(i\omega_0)','-1/k','Location','best');
    axis equal;

    if doExport
        exportgraphics(fig,fullfile(outDir,"02_nyquist.png"),'Resolution',220);
        if doPDF
            exportgraphics(fig,fullfile(outDir,"02_nyquist.pdf"),'ContentType','vector');
        end
    end
end

function plot_phi_amplitude(Agrid,PhiGrid,a0,Phi0,outDir,doExport,doPDF)

    fig = figure('Color','w','Position',[100 100 850 600]);

    plot(Agrid,PhiGrid,'LineWidth',1.2);
    hold on; grid on; box on;

    yline(0,'--','LineWidth',1.0);
    plot(a0,Phi0,'ro','MarkerSize',7,'LineWidth',1.5);

    xlabel('A');
    ylabel('\Phi(A)','Interpreter','tex');
    title('Condicion de amplitud por funcion descriptiva');
    legend('\Phi(A)','0','a_0','Location','best');

    if doExport
        exportgraphics(fig,fullfile(outDir,"03_phi_amplitud.png"),'Resolution',220);
        if doPDF
            exportgraphics(fig,fullfile(outDir,"03_phi_amplitud.pdf"),'ContentType','vector');
        end
    end
end

function plot_eigs_P0(eigP0,omega0,d,outDir,doExport,doPDF)

    fig = figure('Color','w','Position',[100 100 750 600]);

    plot(real(eigP0),imag(eigP0),'ko','MarkerFaceColor','w', ...
        'MarkerSize',8,'LineWidth',1.5);
    hold on; grid on; box on;

    plot(0, omega0,'rx','MarkerSize',10,'LineWidth',2);
    plot(0,-omega0,'rx','MarkerSize',10,'LineWidth',2);
    plot(-d,0,'bs','MarkerSize',8,'LineWidth',1.5);

    xline(0,'--','LineWidth',0.8);
    yline(0,'--','LineWidth',0.8);

    xlabel('Re\{\lambda\}','Interpreter','tex');
    ylabel('Im\{\lambda\}','Interpreter','tex');
    title('Autovalores de P_0 y polos canonicos','Interpreter','tex');
    legend('eig(P_0)','\pm i\omega_0','-d','Location','best');
    axis equal;

    if doExport
        exportgraphics(fig,fullfile(outDir,"04_autovalores_P0.png"),'Resolution',220);
        if doPDF
            exportgraphics(fig,fullfile(outDir,"04_autovalores_P0.pdf"),'ContentType','vector');
        end
    end
end

function plot_harmonic_seed(t,XHar,omega0,a0,outDir,doExport,doPDF)

    fig = figure('Color','w','Position',[100 100 1150 750]);
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    plot(t,XHar(1,:),'LineWidth',1.1); hold on; grid on; box on;
    plot(t,XHar(2,:),'LineWidth',1.1);
    plot(t,XHar(3,:),'LineWidth',1.1);
    xlabel('t'); ylabel('X_h(t)');
    title('Semilla armonica temporal');
    legend('x','y','z','Location','best');

    nexttile;
    plot3(XHar(1,:),XHar(2,:),XHar(3,:),'LineWidth',1.2);
    grid on; box on;
    xlabel('x'); ylabel('y'); zlabel('z');
    title('Semilla armonica en espacio de fase');
    view(35,25);

    nexttile;
    plot(XHar(1,:),XHar(2,:),'LineWidth',1.2);
    grid on; box on;
    xlabel('x'); ylabel('y');
    title('Proyeccion x-y');

    nexttile;
    plot(XHar(1,:),XHar(3,:),'LineWidth',1.2);
    grid on; box on;
    xlabel('x'); ylabel('z');
    title('Proyeccion x-z');

    sgtitle(sprintf('Semilla armonica: omega0 = %.6f, a0 = %.6f',omega0,a0));

    if doExport
        exportgraphics(fig,fullfile(outDir,"05_semilla_armonica.png"),'Resolution',220);
        if doPDF
            exportgraphics(fig,fullfile(outDir,"05_semilla_armonica.pdf"),'ContentType','vector');
        end
    end
end

function plot_integer_comparison(tLin,XLin,tOrg,XOrg,X0,outDir,doExport,doPDF)

    XLinI = interp1(tLin,XLin,tOrg,'pchip');
    E = XOrg - XLinI;
    errAbs = sqrt(sum(E.^2,2));
    errRel = errAbs ./ max(sqrt(sum(XOrg.^2,2)),1e-12);

    fprintf('\nResumen simulacion entera:\n');
    fprintf('Error absoluto final = %.6e\n',errAbs(end));
    fprintf('Error relativo final = %.6e\n',errRel(end));
    fprintf('Error relativo max   = %.6e\n',max(errRel));

    fig = figure('Color','w','Position',[100 100 1300 800]);
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    plot(tOrg,XOrg(:,1),'LineWidth',1.1); hold on; grid on; box on;
    plot(tLin,XLin(:,1),'--','LineWidth',1.1);
    xlabel('t'); ylabel('x(t)');
    title('Componente x');
    legend('Original','Lineal P_0','Location','best');

    nexttile;
    plot(tOrg,XOrg(:,2),'LineWidth',1.1); hold on; grid on; box on;
    plot(tLin,XLin(:,2),'--','LineWidth',1.1);
    xlabel('t'); ylabel('y(t)');
    title('Componente y');
    legend('Original','Lineal P_0','Location','best');

    nexttile;
    plot(tOrg,XOrg(:,3),'LineWidth',1.1); hold on; grid on; box on;
    plot(tLin,XLin(:,3),'--','LineWidth',1.1);
    xlabel('t'); ylabel('z(t)');
    title('Componente z');
    legend('Original','Lineal P_0','Location','best');

    nexttile;
    plot3(XOrg(:,1),XOrg(:,2),XOrg(:,3),'LineWidth',1.1);
    hold on; grid on; box on;
    plot3(XLin(:,1),XLin(:,2),XLin(:,3),'--','LineWidth',1.1);
    plot3(X0(1),X0(2),X0(3),'ko','MarkerFaceColor','w','LineWidth',1.3);
    xlabel('x'); ylabel('y'); zlabel('z');
    title('Retrato de fase');
    legend('Original','Lineal P_0','X(0)','Location','best');
    view(35,25);

    if doExport
        exportgraphics(fig,fullfile(outDir,"06_comparacion_original_lineal.png"),'Resolution',220);
        if doPDF
            exportgraphics(fig,fullfile(outDir,"06_comparacion_original_lineal.pdf"),'ContentType','vector');
        end
    end

    figErr = figure('Color','w','Position',[100 100 850 550]);
    semilogy(tOrg,errRel,'LineWidth',1.2);
    grid on; box on;
    xlabel('t');
    ylabel('||X_{orig}-X_{lin}||/||X_{orig}||','Interpreter','tex');
    title('Error relativo entre sistema original y aproximacion lineal');

    if doExport
        exportgraphics(figErr,fullfile(outDir,"07_error_relativo_original_lineal.png"),'Resolution',220);
        if doPDF
            exportgraphics(figErr,fullfile(outDir,"07_error_relativo_original_lineal.pdf"),'ContentType','vector');
        end
    end
end