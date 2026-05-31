function result = run_dk2018_fde12_oracle(ne, ext_fcn, t_start, h_norm, t_end, x_start, h, q, output_csv)
%RUN_DK2018_FDE12_ORACLE Execute the DK2018 block-restart contract with FDE12.
%
% This runner mirrors the numerical loop in FO_Lyapunov.m while omitting
% plotting and console output. fde12 and the extended-system function must
% already be available on the MATLAB path.

if exist('fde12', 'file') ~= 2
    error('MATLAB:run_dk2018_fde12_oracle:MissingFDE12', ...
        'fde12.m must be available on the MATLAB path.');
end

x = zeros(ne * (ne + 1), 1);
x0 = x;
c = zeros(ne, 1);
gsc = c;
zn = c;
n_it = round((t_end - t_start) / h_norm);
convergence = zeros(n_it, ne + 1);

x(1:ne) = x_start;
for i = 1:ne
    x((ne + 1) * i) = 1.0;
end
t = t_start;

for it = 1:n_it
    % The 2025 fde12 revision prints an accidental ans value on every call.
    % Capture solver console output without modifying the external source.
    evalc('[~, Y] = fde12(q, ext_fcn, t, t + h_norm, x, h);');
    t = t + h_norm;
    Y = transpose(Y);
    x = Y(size(Y, 1), :);

    for i = 1:ne
        for j = 1:ne
            x0(ne * i + j) = x(ne * j + i);
        end
    end

    zn(1) = 0.0;
    for j = 1:ne
        zn(1) = zn(1) + x0(ne * j + 1)^2;
    end
    zn(1) = sqrt(zn(1));
    for j = 1:ne
        x0(ne * j + 1) = x0(ne * j + 1) / zn(1);
    end

    for j = 2:ne
        for k = 1:j - 1
            gsc(k) = 0.0;
            for l = 1:ne
                gsc(k) = gsc(k) + x0(ne * l + j) * x0(ne * l + k);
            end
        end
        for k = 1:ne
            for l = 1:j - 1
                x0(ne * k + j) = x0(ne * k + j) - gsc(l) * x0(ne * k + l);
            end
        end
        zn(j) = 0.0;
        for k = 1:ne
            zn(j) = zn(j) + x0(ne * k + j)^2;
        end
        zn(j) = sqrt(zn(j));
        for k = 1:ne
            x0(ne * k + j) = x0(ne * k + j) / zn(j);
        end
    end

    c = c + log(zn);
    LE = c / (t - t_start);
    convergence(it, :) = [t, transpose(LE)];

    for i = 1:ne
        for j = 1:ne
            x(ne * j + i) = x0(ne * i + j);
        end
    end
    x = transpose(x);
end

if nargin >= 9 && ~isempty(output_csv)
    output_dir = fileparts(output_csv);
    if ~isempty(output_dir) && ~isfolder(output_dir)
        mkdir(output_dir);
    end
    writematrix(convergence, output_csv);
end

result = struct( ...
    'time', t, ...
    'exponents', transpose(LE), ...
    'final_state', transpose(x(1:ne)), ...
    'iterations', n_it);
end
