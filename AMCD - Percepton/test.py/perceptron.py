import numpy as np
import matplotlib.pyplot as plt
import os



def carregar_dados(caminho_train, caminho_test, digito_positivo=1):

    dados_train = np.loadtxt(caminho_train)
    dados_test  = np.loadtxt(caminho_test)

    X_train = dados_train[:, 1:]
    X_test  = dados_test[:, 1:]

    y_train = np.where(dados_train[:, 0] == digito_positivo, +1, -1).astype(float)
    y_test  = np.where(dados_test[:, 0]  == digito_positivo, +1, -1).astype(float)

    return X_train, y_train, X_test, y_test


class PocketPLA:


    def __init__(self, max_iter=10000, seed=42):
        self.max_iter = max_iter
        self.seed = seed
        self.w = None              
        self.w_pocket = None        
        self.erro_pocket = np.inf    
        self.historico_erros = []   

    def _adicionar_bias(self, X):

        N = X.shape[0]
        return np.hstack([np.ones((N, 1)), X])

    def _calcular_erro(self, X_b, y, w):

        predicoes = np.sign(X_b @ w)
        return np.mean(predicoes != y)

    def treinar(self, X, y):

        rng = np.random.default_rng(self.seed)

        # Adiciona bias
        X_b = self._adicionar_bias(X)
        N, d = X_b.shape

        self.w = np.zeros(d)
        self.w_pocket = self.w.copy()
        self.erro_pocket = self._calcular_erro(X_b, y, self.w)
        self.historico_erros = [self.erro_pocket]

        for t in range(self.max_iter):

            predicoes = np.sign(X_b @ self.w)


            errados = np.where(predicoes != y)[0]

            if len(errados) == 0:
                self.w_pocket = self.w.copy()
                self.erro_pocket = 0.0
                self.historico_erros.append(0.0)
                print(f"  Convergiu na iteração {t}! Erro in-sample = 0")
                break

            idx = rng.choice(errados)
            x_n = X_b[idx]
            y_n = y[idx]

            self.w = self.w + y_n * x_n

            erro_atual = self._calcular_erro(X_b, y, self.w)
            if erro_atual < self.erro_pocket:
                self.w_pocket = self.w.copy()
                self.erro_pocket = erro_atual

            self.historico_erros.append(self.erro_pocket)
        else:
            print(f"  Parou após {self.max_iter} iterações.")
            print(f"  Melhor erro in-sample (pocket) = {self.erro_pocket:.4f}")

        return self

    def predizer(self, X):
        X_b = self._adicionar_bias(X)
        return np.sign(X_b @ self.w_pocket)

    def calcular_erro_teste(self, X_test, y_test):

        predicoes = self.predizer(X_test)
        return np.mean(predicoes != y_test)


def plotar_fronteira_decisao(modelo, X, y, titulo, nome_arquivo):

    w = modelo.w_pocket
    fig, ax = plt.subplots(figsize=(10, 7))

    pos = y == +1
    neg = y == -1

    ax.scatter(X[pos, 0], X[pos, 1],
               c='#2196F3', marker='o', s=20, alpha=0.6, label='Classe +1 (dígito alvo)')
    ax.scatter(X[neg, 0], X[neg, 1],
               c='#F44336', marker='x', s=20, alpha=0.4, label='Classe -1 (outros)')

    if abs(w[2]) > 1e-10:
        x1_min, x1_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
        x1_vals = np.linspace(x1_min, x1_max, 300)
        x2_vals = -(w[0] + w[1] * x1_vals) / w[2]

        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        mascara = (x2_vals >= x2_min) & (x2_vals <= x2_max)
        ax.plot(x1_vals[mascara], x2_vals[mascara],
                'k-', linewidth=2.5, label='Fronteira de decisão')

    ax.set_xlabel('Feature 1 (Intensidade)', fontsize=12)
    ax.set_ylabel('Feature 2 (Simetria)', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot salvo em: {nome_arquivo}")


def plotar_evolucao_erro(historico, nome_arquivo):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(historico, color='#4CAF50', linewidth=1.5)
    ax.set_xlabel('Iteração', fontsize=12)
    ax.set_ylabel('Erro In-Sample (Pocket)', fontsize=12)
    ax.set_title('Evolução do Erro do Pocket PLA', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot salvo em: {nome_arquivo}")


def main():
    CAMINHO_TRAIN = "digits.train"
    CAMINHO_TEST  = "digits.test"

    for caminho in [CAMINHO_TRAIN, CAMINHO_TEST]:
        if not os.path.exists(caminho):
            print(f"ERRO: arquivo '{caminho}' não encontrado!")
            print("Coloque digits.train e digits.test na mesma pasta do script.")
            return

    print("=" * 65)
    print("  POCKET PLA — Implementação do Perceptron")
    print("=" * 65)

    digito = 1
    print(f"\n{'─' * 65}")
    print(f"  Classificação: dígito {digito} vs todos os outros")
    print(f"{'─' * 65}")

    X_train, y_train, X_test, y_test = carregar_dados(
        CAMINHO_TRAIN, CAMINHO_TEST, digito_positivo=digito
    )

    n_pos_train = np.sum(y_train == +1)
    n_neg_train = np.sum(y_train == -1)
    print(f"  Treino: {len(y_train)} amostras "
          f"({n_pos_train} positivas, {n_neg_train} negativas)")
    print(f"  Teste:  {len(y_test)} amostras")

    # Treina o Pocket PLA
    print(f"\n  Treinando Pocket PLA (max_iter=10000)...")
    modelo = PocketPLA(max_iter=10000, seed=42)
    modelo.treinar(X_train, y_train)

    # Erro in-sample (treino)
    erro_in = modelo.erro_pocket
    print(f"\n  Erro In-Sample  (E_in):  {erro_in:.4f}  "
          f"({erro_in*100:.2f}%)")

    # Erro out-of-sample (teste)
    erro_out = modelo.calcular_erro_teste(X_test, y_test)
    print(f"  Erro Out-Sample (E_out): {erro_out:.4f}  "
          f"({erro_out*100:.2f}%)")

    # Acurácia
    print(f"\n  Acurácia no treino: {(1 - erro_in)*100:.2f}%")
    print(f"  Acurácia no teste:  {(1 - erro_out)*100:.2f}%")

    # Pesos encontrados
    w = modelo.w_pocket
    print(f"\n  Pesos (pocket): w0={w[0]:.6f}, w1={w[1]:.6f}, w2={w[2]:.6f}")

    # --- Plots ---
    print(f"\n  Gerando plots...")

    plotar_fronteira_decisao(
        modelo, X_train, y_train,
        titulo=f"Fronteira de Decisão — Dígito {digito} vs Resto (Treino)",
        nome_arquivo=f"fronteira_digito_{digito}_treino.png"
    )

    plotar_fronteira_decisao(
        modelo, X_test, y_test,
        titulo=f"Fronteira de Decisão — Dígito {digito} vs Resto (Teste)",
        nome_arquivo=f"fronteira_digito_{digito}_teste.png"
    )

    plotar_evolucao_erro(
        modelo.historico_erros,
        nome_arquivo=f"evolucao_erro_digito_{digito}.png"
    )

    print(f"\n{'=' * 65}")
    print("  COMPARAÇÃO: todos os dígitos (cada um vs resto)")
    print(f"{'=' * 65}")
    print(f"  {'Dígito':>8} | {'E_in':>10} | {'E_out':>10} | {'Acurácia':>10}")
    print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}")

    for d in range(10):
        X_tr, y_tr, X_te, y_te = carregar_dados(
            CAMINHO_TRAIN, CAMINHO_TEST, digito_positivo=d
        )
        m = PocketPLA(max_iter=5000, seed=42)
        m.treinar(X_tr, y_tr)
        e_in  = m.erro_pocket
        e_out = m.calcular_erro_teste(X_te, y_te)
        print(f"  {d:>8} | {e_in:>10.4f} | {e_out:>10.4f} | "
              f"{(1-e_out)*100:>9.2f}%")

    print(f"\n{'=' * 65}")
    print("  Concluído! Verifique os plots gerados na pasta.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()