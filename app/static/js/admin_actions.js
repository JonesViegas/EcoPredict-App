// --- FUNÇÃO DE NOTIFICAÇÃO (TOAST) ---
function showToast(message, type = 'info') {
    const toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        alert(`[${type.toUpperCase()}] ${message}`);
        return;
    }
    const toastId = 'toast-' + Date.now();
    const bg_color = { success: 'bg-success', danger: 'bg-danger', warning: 'bg-warning', info: 'bg-info' }[type] || 'bg-secondary';
    const icon = { success: 'fa-check-circle', danger: 'fa-exclamation-triangle', warning: 'fa-exclamation-circle', info: 'fa-info-circle' }[type] || 'fa-bell';
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white ${bg_color} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body"><i class="fas ${icon} me-2"></i>${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = document.getElementById(toastId);
    if (typeof bootstrap !== 'undefined') {
        const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
        toast.show();
        toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
    }
}

// --- FUNÇÃO GENÉRICA PARA CHAMAR A API DE ADMIN ---
async function performAdminAction(url, options = {}, confirmMessage, successMessage) {
    if (confirmMessage && !confirm(confirmMessage)) return;
    showToast(options.processingMessage || 'Processando sua solicitação...', 'info');
    
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    try {
        const response = await fetch(url, {
            method: options.method || 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken, ...options.headers },
            body: options.body ? JSON.stringify(options.body) : null
        });
        const result = await response.json();
        if (response.ok && result.success) {
            showToast(result.message || successMessage, 'success');
            return result;
        } else {
            throw new Error(result.error || 'Ocorreu um erro desconhecido no servidor.');
        }
    } catch (error) {
        console.error(`Erro na ação ${url}:`, error);
        showToast(`Falha na operação: ${error.message}`, 'danger');
    }
}

// --- FUNÇÕES DE AÇÃO ESPECÍFICAS (CONECTADAS AO HTML 'onclick') ---
function backupSystem() {
    performAdminAction('/admin/api/backup', { processingMessage: 'Iniciando backup...' }, 'Deseja criar um backup completo do sistema agora?', 'Backup criado com sucesso!');
}
function clearSystemCache() {
    performAdminAction('/admin/api/clear_cache', { processingMessage: 'Limpando cache...' }, 'Limpar o cache do sistema?', 'Cache do sistema limpo com sucesso!');
}
function optimizeDatabase() {
    performAdminAction('/admin/api/optimize_db', { processingMessage: 'Otimizando banco de dados...' }, 'Otimizar o banco de dados?', 'Banco de dados otimizado com sucesso!');
}
function restartSystem() {
    performAdminAction('/admin/api/restart_system', { processingMessage: 'Enviando comando de reinicialização...' }, 'TEM CERTEZA que deseja reiniciar o sistema?', 'Sistema será reiniciado em breve.');
}
function downloadLogs() {
    showToast('Preparando download dos logs...', 'info');
    window.location.href = '/admin/api/download_logs';
}
function clearLogs() {
    performAdminAction('/admin/api/clear_old_logs', { processingMessage: 'Limpando logs antigos...' }, 'Limpar logs com mais de 30 dias?', 'Logs antigos removidos com sucesso!');
}
function toggleMaintenanceModeGlobal() {
    const checkbox = document.getElementById('maintenanceModeGlobal');
    const isChecked = checkbox.checked;
    const message = isChecked ? 'Ativar o modo manutenção?' : 'Desativar o modo manutenção?';
    if (confirm(message)) {
        performAdminAction('/admin/api/toggle_maintenance', { body: { maintenance_mode: isChecked } }, null, 'Modo manutenção atualizado!');
    } else {
        checkbox.checked = !isChecked;
    }
}