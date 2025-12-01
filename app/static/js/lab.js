document.addEventListener('DOMContentLoaded', function() {
    console.log('Laboratório de Análise Carregado.');
});

// ===============================================
//          NOVAS FUNÇÕES AUXILIARES
// ===============================================
async function loadDatasetColumns() {
        try {
            const response = await fetch(`/api/dataset-features/${DATASET_ID}`);
            if (!response.ok) throw new Error('Falha ao carregar colunas do dataset.');

            const data = await response.json();
            if (data.success) {
                populateSelects(data.columns);
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            alert(`Erro: ${error.message}`);
        }
    }

    // Função para preencher todas as listas de seleção
    function populateSelects(columns) {
        // Encontra todos os selects que precisam ser populados
        const selects = {
            clustering: document.getElementById('clustering-features'),
            pca: document.getElementById('pca-features'),
            classificationTarget: document.getElementById('classification-target'),
            classificationFeatures: document.getElementById('classification-features'),
            regressionTarget: document.getElementById('regression-target'),
            regressionFeatures: document.getElementById('regression-features')
        };

        // Limpa todas as opções existentes
        for (const key in selects) {
            if (selects[key]) selects[key].innerHTML = '';
        }

        columns.forEach(col => {
            const option = new Option(col.name, col.name);

            // Adiciona a opção aos selects apropriados com base no tipo
            if (col.type === 'numeric') {
                selects.clustering.add(option.cloneNode(true));
                selects.pca.add(option.cloneNode(true));
                selects.regressionTarget.add(option.cloneNode(true));
            }
            if (col.type === 'categorical') {
                selects.classificationTarget.add(option.cloneNode(true));
            }
            
            // Todas as colunas podem ser features de entrada para classificação e regressão
            selects.classificationFeatures.add(option.cloneNode(true));
            selects.regressionFeatures.add(option.cloneNode(true));
        });
    }

    loadDatasetColumns();


/**
 * Imprime o conteúdo da área de resultados.
 */
function printReport() {
    const resultsArea = document.getElementById('results-area');
    const originalContents = document.body.innerHTML;
    const printContents = resultsArea.innerHTML;
    
    document.body.innerHTML = `
        <html>
            <head>
                <title>Relatório de Análise - EcoPredict</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { padding: 2rem; }
                    img { max-width: 100%; height: auto; border: 1px solid #dee2e6; }
                    .table { font-size: 0.9rem; }
                </style>
            </head>
            <body>
                <h1>Relatório de Análise</h1>
                <hr>
                ${printContents}
            </body>
        </html>`;
    
    window.print();
    
    document.body.innerHTML = originalContents;
    // Precisamos recarregar os listeners de eventos, a forma mais simples é recarregar a página
    window.location.reload();
}

/**
 * Adiciona um botão de download a um elemento de imagem.
 * @param {string} plotContainerId - O ID do container onde o gráfico está.
 * @param {string} chartTitle - O título para o nome do arquivo.
 */
function addDownloadButton(plotContainerId, chartTitle) {
    const plotContainer = document.getElementById(plotContainerId);
    const plotImage = plotContainer.querySelector('img');
    if (!plotImage) return;

    const downloadLink = document.createElement('a');
    downloadLink.href = plotImage.src;
    downloadLink.download = `${chartTitle.replace(/\s+/g, '_').toLowerCase()}.png`;
    downloadLink.className = 'btn btn-secondary btn-sm mt-2';
    downloadLink.innerHTML = '<i class="fas fa-download me-2"></i>Baixar Gráfico';

    plotContainer.appendChild(downloadLink);
}


// ===============================================
//         FUNÇÕES PRINCIPAIS (EXISTENTES)
// ===============================================

function getSelectedOptions(selectId) {
    const select = document.getElementById(selectId);
    return Array.from(select.selectedOptions).map(option => option.value);
}

async function runAnalysis(analysisType) {
    const resultsArea = document.getElementById('results-area');
    resultsArea.innerHTML = `<div class="d-flex justify-content-center align-items-center h-100"><div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;"></div><span class="ms-3">Executando análise...</span></div>`;
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    let payload = { dataset_id: DATASET_ID };
    let url = '';

    try {
        switch (analysisType) {
            case 'clustering':
                url = '/lab/run_clustering';
                payload.features = getSelectedOptions('clustering-features');
                payload.n_clusters = document.getElementById('clustering-k').value;
                if (payload.features.length < 2) throw new Error("Selecione pelo menos 2 features.");
                break;
            case 'pca':
                url = '/lab/run_pca';
                payload.features = getSelectedOptions('pca-features');
                if (payload.features.length < 2) throw new Error("Selecione pelo menos 2 features.");
                break;
            case 'classification':
                url = '/lab/run_classification';
                payload.features = getSelectedOptions('classification-features');
                payload.target = document.getElementById('classification-target').value;
                if (payload.features.length < 1) throw new Error("Selecione pelo menos 1 feature de entrada.");
                break;
            case 'regression':
                url = '/lab/run_regression';
                payload.features = getSelectedOptions('regression-features');
                payload.target = document.getElementById('regression-target').value;
                if (payload.features.length < 1) throw new Error("Selecione pelo menos 1 feature de entrada.");
                break;
            default: throw new Error('Tipo de análise desconhecido.');
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `Erro do Servidor: Status ${response.status}` }));
            throw new Error(errorData.error);
        }

        const results = await response.json();
        if (results.error) {
            throw new Error(results.error);
        }
        
        displayResults(analysisType, results);

    } catch (error) {
        resultsArea.innerHTML = `<div class="alert alert-danger"><strong>Erro:</strong> ${error.message}</div>`;
    }
}

// ===============================================
//          FUNÇÃO DISPLAY ATUALIZADA
// ===============================================

function displayResults(analysisType, results) {
    const resultsArea = document.getElementById('results-area');
    let html = '';

    switch (analysisType) {
        case 'clustering':
            html = `
                <h4>Análise de Clusterização (K-Means)</h4>
                <p>O gráfico abaixo mostra os grupos (clusters) encontrados nos seus dados. Pontos com a mesma cor pertencem ao mesmo grupo.</p>
                <div id="plot-container-clustering" class="text-center">
                    <img src="data:image/png;base64,${results.plot}" class="img-fluid border rounded mb-3" alt="Gráfico de Clusterização">
                </div>
                <h6>Estatísticas dos Clusters</h6>
                ${results.stats_table}`;
            resultsArea.innerHTML = html;
            addDownloadButton('plot-container-clustering', 'grafico_clusterizacao');
            break;
        case 'pca':
             html = `
                <h4>Análise de Componentes Principais (PCA)</h4>
                <p>PCA ajuda a simplificar dados complexos, criando novos eixos (componentes) que capturam o máximo de informação. Use os gráficos abaixo para entender o resultado.</p>
                
                <hr>
                
                <div class="row mb-4">
                    <div class="col-md-6 text-center" id="plot-container-pca-1">
                        <h6>1. Quanta informação cada componente captura?</h6>
                        <img src="data:image/png;base64,${results.scree_plot}" class="img-fluid border rounded" alt="Scree Plot">
                    </div>
                    <div class="col-md-6 text-center" id="plot-container-pca-2">
                        <h6>2. Como os dados se parecem nos novos eixos?</h6>
                        <img src="data:image/png;base64,${results.components_plot}" class="img-fluid border rounded" alt="Components Plot">
                    </div>
                </div>

                <hr>

                <div id="plot-container-pca-3" class="text-center">
                    <h6>3. O que cada Componente Principal significa?</h6>
                    <p class="text-muted">Este mapa de calor é a chave para a interpretação. Ele mostra o "peso" de cada feature original (no eixo Y) em cada novo Componente Principal (no eixo X). <strong>Valores altos (positivos em azul ou negativos em vermelho) indicam que a feature é muito importante para aquele componente.</strong></p>
                    <img src="data:image/png;base64,${results.loadings_heatmap}" class="img-fluid border rounded" alt="PCA Loadings Heatmap">
                </div>
                `;
            resultsArea.innerHTML = html;
            addDownloadButton('plot-container-pca-1', 'grafico_variancia_pca');
            addDownloadButton('plot-container-pca-2', 'grafico_componentes_pca');
            addDownloadButton('plot-container-pca-3', 'grafico_contribuicoes_pca'); // <-- Adiciona botão de download para o novo gráfico
            break;
        case 'classification':
            html = `
                <h4>Avaliação do Modelo de Classificação</h4>
                <p>O modelo foi treinado para prever a variável <strong>'${document.getElementById('classification-target').value}'</strong>. A acurácia geral foi de <strong>${results.accuracy}</strong>.</p>
                <h6>Matriz de Confusão</h6>
                <p>Mostra os acertos e erros do modelo. A diagonal principal representa as previsões corretas.</p>
                <div id="plot-container-classification" class="text-center">
                    <img src="data:image/png;base64,${results.confusion_matrix_plot}" class="img-fluid border rounded mb-3" alt="Matriz de Confusão">
                </div>
                <h6>Relatório de Classificação</h6>
                <div class="table-responsive">${results.report_table}</div>`;
            resultsArea.innerHTML = html;
            addDownloadButton('plot-container-classification', 'matriz_confusao');
            break;
        case 'regression':
             html = `
                <h4>Avaliação do Modelo de Regressão</h4>
                <p>O modelo foi treinado para prever o valor de <strong>'${document.getElementById('regression-target').value}'</strong>.</p>
                <ul class="list-group list-group-flush mb-3">
                    <li class="list-group-item"><strong>R-squared (R²):</strong> ${results.r2_score} (Quanto mais perto de 1, melhor o modelo explica os dados)</li>
                    <li class="list-group-item"><strong>RMSE (Erro):</strong> ${results.rmse} (Quanto menor, mais precisas são as previsões)</li>
                </ul>
                <h6>Valores Reais vs. Previstos</h6>
                <p>Quanto mais próximos os pontos estiverem da linha vermelha, melhores são as previsões.</p>
                <div id="plot-container-regression" class="text-center">
                    <img src="data:image/png;base64,${results.prediction_plot}" class="img-fluid border rounded" alt="Previsão vs Real">
                </div>`;
            resultsArea.innerHTML = html;
            addDownloadButton('plot-container-regression', 'grafico_regressao');
            break;
    }
}