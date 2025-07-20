% Parâmetros do problema
L = 1.0;               % comprimento da barra (m)
N = 100;                % número de graus de liberdade
rho = 1.0;             % densidade linear (kg/m)
G = 1.0;               % módulo de cisalhamento (Pa)
J = 1.0;               % momento polar de inércia (m^4)

dx = L / (N + 1);      % comprimento de cada elemento

% Montagem da matriz de rigidez torcional [Kθ]
k0 = G * J / dx;
K = zeros(N, N);
for i = 1:N
    if i > 1
        K(i, i-1) = -k0;
    end
    K(i, i) = 2*k0;
    if i < N
        K(i, i+1) = -k0;
    end
end

% Montagem da matriz de inércia [I]
I_rot = rho * dx;        % inércia rotacional de cada elemento
I = I_rot * eye(N);      % matriz diagonal

% Resolvendo o problema generalizado de autovalor
[Phi, D] = eig(K, I);
omegas = sqrt(diag(D));   % frequências naturais rad/s
f_numerico = omegas / (2*pi);  % frequências em Hz

% Solução exata contínua
n_modes = min(5, N);
omega_analitico = (1:n_modes)' * pi;
f_analitico = omega_analitico / (2*pi);

% Exibindo os resultados
disp('Modo   omega_numérico (rad/s)   f_numérico (Hz)   omega_analítico (rad/s)   f_analítico (Hz)');
for i = 1:n_modes
    fprintf('%4d   %20.6f   %14.6f   %23.6f   %16.6f\n', ...
        i, omegas(i), f_numerico(i), omega_analitico(i), f_analitico(i));
end

% Plotando os primeiros modos de vibração
for mode = 1:n_modes
    figure;
    plot(linspace(dx, L-dx, N), Phi(:, mode), '-o');
    title(['Modo ', num2str(mode), ...
        ' - \omega = ', num2str(omegas(mode), '%.4f'), ' rad/s']);
    xlabel('Comprimento da barra');
    ylabel('\theta');
    grid on;
end

