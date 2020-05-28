{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Transformações em dados heterogeneos.ipynb",
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zlvnMZcyyNwi",
        "colab_type": "text"
      },
      "source": [
        "# Objetivo\n",
        "\n",
        "A ideia deste notebook é demonstrar como podemos fazer transformações em dados heterogêneos com o sklearn. A ideia é facilitar a manipulação de tais tipos de dados, que são comuns no dia-a-dia\n",
        "\n",
        "### Gerando os dados"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gR8EeFcPxHw4",
        "colab_type": "code",
        "outputId": "17482770-14fc-4e87-a1ce-af2ad04c207f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        }
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "# Dados de exemplo\n",
        "dados = pd.DataFrame({\n",
        "    'idade': [15, 23, 19, 30, 44, np.NaN],\n",
        "    'sexo': ['homem', 'mulher', 'homem', 'homem', np.NaN,'homem'],\n",
        "    'altura': [1.6, 1.7, np.NaN, 1.8, 1.75, 1.65],\n",
        "    'classe':  [-1, 1, -1, 1, 1, -1]\n",
        "    })\n",
        "dados"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>idade</th>\n",
              "      <th>sexo</th>\n",
              "      <th>altura</th>\n",
              "      <th>classe</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>15.0</td>\n",
              "      <td>homem</td>\n",
              "      <td>1.60</td>\n",
              "      <td>-1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>23.0</td>\n",
              "      <td>mulher</td>\n",
              "      <td>1.70</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>19.0</td>\n",
              "      <td>homem</td>\n",
              "      <td>NaN</td>\n",
              "      <td>-1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>30.0</td>\n",
              "      <td>homem</td>\n",
              "      <td>1.80</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>44.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.75</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>NaN</td>\n",
              "      <td>homem</td>\n",
              "      <td>1.65</td>\n",
              "      <td>-1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   idade    sexo  altura  classe\n",
              "0   15.0   homem    1.60      -1\n",
              "1   23.0  mulher    1.70       1\n",
              "2   19.0   homem     NaN      -1\n",
              "3   30.0   homem    1.80       1\n",
              "4   44.0     NaN    1.75       1\n",
              "5    NaN   homem    1.65      -1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nHe5-CStziqB",
        "colab_type": "code",
        "outputId": "fa71c49e-bdf1-4b1f-982f-188e2b4329e5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 208
        }
      },
      "source": [
        "dados.info()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 6 entries, 0 to 5\n",
            "Data columns (total 4 columns):\n",
            " #   Column  Non-Null Count  Dtype  \n",
            "---  ------  --------------  -----  \n",
            " 0   idade   5 non-null      float64\n",
            " 1   sexo    5 non-null      object \n",
            " 2   altura  5 non-null      float64\n",
            " 3   classe  6 non-null      int64  \n",
            "dtypes: float64(2), int64(1), object(1)\n",
            "memory usage: 320.0+ bytes\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "w2wMubZjgew2",
        "colab_type": "text"
      },
      "source": [
        "A partir do comando `info()` nota-se que existem 2 variáveis de entrada numéricas (idade e altura) e 1 variável de entrada categórica (sexo). A variável classe é a variável alvo."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HEsjWPWVzxq8",
        "colab_type": "text"
      },
      "source": [
        "### Transformando os dados\n",
        "\n",
        "Considere que queremos realizar as seguintes operações: \n",
        "\n",
        "\n",
        "*   **Nos dados numéricos:** substituir os dados faltantes das variáveis pela média dos valores presentes. Depois padronizar o intervalo dessas variáveis\n",
        "*   **Nos dados categóricos:** Substituir os valores faltantes pelo valor mais frequente. Depois transformar essas categorias em um atributo numérico para usar no nosso modelo de rede neural\n",
        "\n",
        "Para fazer essas operações em colunas específicas, podemos utilizar as ferramentas `sklearn.pipeline.Pipeline` e `sklearn.compose.ColumnTransformer`.\n",
        "\n",
        "A ferramenta `sklearn.pipeline.Pipeline` cria uma sequencia de transformações nos dados, enquanto que a `sklearn.compose.ColumnTransformer` realiza uma transformação em dados de uma coluna.\n",
        "\n",
        "Para substituir valores faltantes utilizaremos a classe `sklearn.impute.SimpleImputer`, para padronizar os dados `sklearn.preprocessingStandardScaler` e para converter atributos categoricos em valores numéricos `sklearn.preprocessing.OneHotEncoder`.\n",
        "\n",
        "Aplicando nos dados do exemplo acima:\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ejAkeP6r2LUi",
        "colab_type": "code",
        "outputId": "db625545-e0f1-4c90-ffa2-cad5b3cea30b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 121
        }
      },
      "source": [
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.impute import SimpleImputer\n",
        "\n",
        "# Criamos um vetor com o nome das classes desejadas\n",
        "features_numericos = ['idade', 'altura']\n",
        "features_categoricos = ['sexo']\n",
        "\n",
        "# Criando os pipelines\n",
        "pipeline_numerico = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='mean')),\n",
        "    ('scaler', StandardScaler())])\n",
        "\n",
        "pipeline_categorico = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "    ('onehot', OneHotEncoder())])\n",
        "\n",
        "# Criando a transformação do conjunto de dados:\n",
        "transformacao = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('transformacao numerica', pipeline_numerico, features_numericos),\n",
        "        ('transformacao categorica', pipeline_categorico, features_categoricos),        \n",
        "    ])\n",
        "\n",
        "# Aplicando a transformação no dataset:\n",
        "dados_transformados = transformacao.fit_transform(dados)\n",
        "dados_transformados.round(2)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-1.2 , -1.55,  1.  ,  0.  ],\n",
              "       [-0.34, -0.  ,  0.  ,  1.  ],\n",
              "       [-0.77, -0.  ,  1.  ,  0.  ],\n",
              "       [ 0.41,  1.55,  1.  ,  0.  ],\n",
              "       [ 1.91,  0.77,  1.  ,  0.  ],\n",
              "       [ 0.  , -0.77,  1.  ,  0.  ]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6rjhWB9V5TTm",
        "colab_type": "text"
      },
      "source": [
        "Note que a ordem das variáveis (features) mudou em relação ao nosso conjunto inicial. A ordem do conjunto transformado é dada pela ordem de processamento das features. No nosso caso, a ordem das features ficou: \n",
        "\n",
        "(features_numericos, features_categoricos) $\\rightarrow$ (idade, altura, sexo)\n",
        "\n",
        " Note também que a transformação `OneHotEncoder` vai transformar os features categóricos em um vetor unitário, que possui $n$ dimensões, onde $n$ é o número de valores diferentes da categoria (no nosso exemplo, 2 - homem e mulher)\n",
        "\n",
        " Observe que o conjunto não possui mais a variável \"classe\". Isso porque não foi efetuada nenhuma transformação nessa variável.\n",
        "\n",
        " Normalmente para problemas de Machine Learning é comum separar as variáveis de entrada (\"sexo\", \"idade\" e \"altura\") da variável alvo (\"classe\"). Mas se for necessário, podemos facilmente concatenar a variável classe e criar uma estrutura `DataFrame`:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CKowk_z08TLy",
        "colab_type": "code",
        "outputId": "37de8e5f-979c-420a-b613-bccb9dbbd293",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        }
      },
      "source": [
        "dados_transformados_com_classe = np.c_[dados_transformados, dados['classe']]\n",
        "dataframe_processado = pd.DataFrame(dados_transformados_com_classe)\n",
        "dataframe_processado"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "      <th>2</th>\n",
              "      <th>3</th>\n",
              "      <th>4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>-1.204464</td>\n",
              "      <td>-1.549193e+00</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>-0.344132</td>\n",
              "      <td>-3.439900e-15</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>-0.774298</td>\n",
              "      <td>-3.439900e-15</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.408657</td>\n",
              "      <td>1.549193e+00</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1.914237</td>\n",
              "      <td>7.745967e-01</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>-7.745967e-01</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "          0             1    2    3    4\n",
              "0 -1.204464 -1.549193e+00  1.0  0.0 -1.0\n",
              "1 -0.344132 -3.439900e-15  0.0  1.0  1.0\n",
              "2 -0.774298 -3.439900e-15  1.0  0.0 -1.0\n",
              "3  0.408657  1.549193e+00  1.0  0.0  1.0\n",
              "4  1.914237  7.745967e-01  1.0  0.0  1.0\n",
              "5  0.000000 -7.745967e-01  1.0  0.0 -1.0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BlqfnDsu-r4T",
        "colab_type": "text"
      },
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "**Opcional:** para adicionar nomes às variáveis do dataframe:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NAEpN9e_-xCS",
        "colab_type": "code",
        "outputId": "e5b504e5-fb28-467f-ed28-4eac19572cfd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        }
      },
      "source": [
        "nomes = []\n",
        "# Variáveis numéricas não tem alteração de tamanho, logo:\n",
        "nomes = nomes + features_numericos\n",
        "\n",
        "# Para os dados categóricos devemos acessar o transformador:\n",
        "transformacao_categorica = transformacao.transformers_[1]\n",
        "# Depois o pipeline\n",
        "pipeline_categorico = transformacao_categorica[1]\n",
        "# E finalmente o onehot\n",
        "transf_onehot = pipeline_categorico.named_steps['onehot']\n",
        "# Para acessar o nome das variáveis usamos o método get_feature_names()\n",
        "nomes = nomes + (transf_onehot.get_feature_names().tolist())\n",
        "\n",
        "nomes.append(\"classe\")\n",
        "\n",
        "dataframe_processado = pd.DataFrame(data = dados_transformados_com_classe, columns=nomes)\n",
        "dataframe_processado.round(2)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>idade</th>\n",
              "      <th>altura</th>\n",
              "      <th>x0_homem</th>\n",
              "      <th>x0_mulher</th>\n",
              "      <th>classe</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>-1.20</td>\n",
              "      <td>-1.55</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>-0.34</td>\n",
              "      <td>-0.00</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>-0.77</td>\n",
              "      <td>-0.00</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.41</td>\n",
              "      <td>1.55</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1.91</td>\n",
              "      <td>0.77</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>0.00</td>\n",
              "      <td>-0.77</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>-1.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   idade  altura  x0_homem  x0_mulher  classe\n",
              "0  -1.20   -1.55       1.0        0.0    -1.0\n",
              "1  -0.34   -0.00       0.0        1.0     1.0\n",
              "2  -0.77   -0.00       1.0        0.0    -1.0\n",
              "3   0.41    1.55       1.0        0.0     1.0\n",
              "4   1.91    0.77       1.0        0.0     1.0\n",
              "5   0.00   -0.77       1.0        0.0    -1.0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    }
  ]
}