/**
 * Password Guesser - Frontend Application
 */

const API_BASE = '';

// ============== API Client ==============

const api = {
    async get(path) {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        return res.json();
    },

    async post(path, body) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) {
            const detail = await res.text();
            throw new Error(detail);
        }
        return res.json();
    }
};

// ============== State ==============

const state = {
    modelLoaded: false,
    llmConfigured: false,
    generating: false,
    lastFeatures: null
};

// ============== DOM Elements ==============

const els = {
    status: document.getElementById('status'),
    apiKey: document.getElementById('apiKey'),
    apiBase: document.getElementById('apiBase'),
    configureLLM: document.getElementById('configureLLM'),
    targetInfo: document.getElementById('targetInfo'),
    useLLM: document.getElementById('useLLM'),
    parallelExtract: document.getElementById('parallelExtract'),
    extractStages: document.getElementById('extractStages'),
    genMethod: document.getElementById('genMethod'),
    nSamples: document.getElementById('nSamples'),
    temperature: document.getElementById('temperature'),
    topK: document.getElementById('topK'),
    topP: document.getElementById('topP'),
    tempSchedule: document.getElementById('tempSchedule'),
    generateBtn: document.getElementById('generateBtn'),
    extractedFeatures: document.getElementById('extractedFeatures'),
    passwordResults: document.getElementById('passwordResults'),
    passwordCount: document.getElementById('passwordCount'),
    extractTime: document.getElementById('extractTime'),
    genTime: document.getElementById('genTime'),
    cacheHitRate: document.getElementById('cacheHitRate'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toastContainer: document.getElementById('toastContainer')
};

// ============== Toast Notifications ==============

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    els.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============== Loading ==============

function showLoading(text = '处理中...') {
    els.loadingText.textContent = text;
    els.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    els.loadingOverlay.classList.add('hidden');
}

// ============== Status ==============

async function checkStatus() {
    try {
        const data = await api.get('/api/status');
        const statusEl = els.status;
        statusEl.className = 'status connected';
        statusEl.querySelector('.status-text').textContent =
            `${data.device} | 模型${data.model_loaded ? '已加载' : '未加载'}`;
        state.modelLoaded = data.model_loaded;
    } catch (e) {
        const statusEl = els.status;
        statusEl.className = 'status error';
        statusEl.querySelector('.status-text').textContent = '连接失败';
    }
}

// ============== LLM Configuration ==============

els.configureLLM.addEventListener('click', async () => {
    const apiKey = els.apiKey.value.trim();
    const apiBase = els.apiBase.value.trim();

    if (!apiKey) {
        showToast('请输入 API Key', 'warning');
        return;
    }

    try {
        showLoading('配置 LLM...');
        await api.post(`/api/configure_llm?api_key=${encodeURIComponent(apiKey)}&api_base=${encodeURIComponent(apiBase)}`);
        state.llmConfigured = true;
        showToast('LLM 配置成功', 'success');
    } catch (e) {
        showToast(`配置失败: ${e.message}`, 'error');
    } finally {
        hideLoading();
    }
});

// ============== Method Description ==============

const methodDescriptions = {
    sampling: '标准采样，支持温度和 top-k/top-p 调节',
    beam: 'Beam Search - 搜索多条路径选最优',
    diverse_beam: '多组 Beam Search，加入多样性惩罚',
    typical: '基于信息熵的典型采样，输出更连贯',
    contrastive: '对比搜索，惩罚重复内容'
};

els.genMethod.addEventListener('change', () => {
    const desc = methodDescriptions[els.genMethod.value] || '';
    let descEl = document.querySelector('.method-desc');
    if (!descEl) {
        descEl = document.createElement('div');
        descEl.className = 'method-desc';
        els.genMethod.parentElement.appendChild(descEl);
    }
    descEl.textContent = desc;
});

// ============== Feature Display ==============

function renderFeatures(features) {
    if (!features) {
        els.extractedFeatures.innerHTML = '<p class="placeholder">无提取结果</p>';
        return;
    }

    const fields = [
        { key: 'full_name', label: '全名' },
        { key: 'first_name', label: '名' },
        { key: 'last_name', label: '姓' },
        { key: 'nickname', label: '昵称' },
        { key: 'birthday', label: '生日' },
        { key: 'phone', label: '电话' },
        { key: 'email_prefix', label: '邮箱前缀' },
        { key: 'city', label: '城市' },
        { key: 'country', label: '国家' },
    ];

    const listFields = [
        { key: 'hobbies', label: '兴趣爱好' },
        { key: 'favorite_words', label: '喜爱词汇' },
        { key: 'favorite_numbers', label: '喜爱数字' },
        { key: 'pet_names', label: '宠物名' },
        { key: 'keywords', label: '关键词' },
    ];

    let html = '';

    // String fields
    for (const field of fields) {
        const value = features[field.key] || '';
        const hasValue = value.length > 0;
        html += `
            <div class="feature-tag ${hasValue ? 'has-value' : ''}">
                <span class="feature-label">${field.label}</span>
                <span class="feature-value ${!hasValue ? 'empty' : ''}">${hasValue ? escapeHtml(value) : '-'}</span>
            </div>
        `;
    }

    // List fields
    for (const field of listFields) {
        const values = features[field.key] || [];
        const hasValue = values.length > 0;
        html += `
            <div class="feature-tag ${hasValue ? 'has-value' : ''}" style="grid-column: span 2;">
                <span class="feature-label">${field.label}</span>
                <span class="feature-value ${!hasValue ? 'empty' : ''}">
                    ${hasValue ? values.map(v => `<span class="feature-list-item">${escapeHtml(String(v))}</span>`).join(' ') : '-'}
                </span>
            </div>
        `;
    }

    els.extractedFeatures.innerHTML = html;
}

// ============== Password Display ==============

function renderPasswords(passwords) {
    if (!passwords || passwords.length === 0) {
        els.passwordResults.innerHTML = '<p class="placeholder">无生成结果</p>';
        els.passwordCount.textContent = '';
        return;
    }

    els.passwordCount.textContent = passwords.length;

    let html = '';
    for (let i = 0; i < passwords.length; i++) {
        const p = passwords[i];
        const scoreText = p.score != null ? p.score.toFixed(3) : '';
        const methodText = p.method || '';
        html += `
            <div class="password-card" onclick="copyPassword(this, '${escapeHtml(p.password)}')" title="点击复制">
                <span class="pwd-text">${escapeHtml(p.password)}</span>
                <div class="pwd-info">
                    ${methodText ? `<span class="pwd-method">${methodText}</span>` : ''}
                    ${scoreText ? `<span class="pwd-score">${scoreText}</span>` : ''}
                </div>
            </div>
        `;
    }

    els.passwordResults.innerHTML = html;
}

function copyPassword(el, password) {
    navigator.clipboard.writeText(password).then(() => {
        el.classList.add('copied');
        showToast(`已复制: ${password}`, 'success', 1500);
        setTimeout(() => el.classList.remove('copied'), 800);
    }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = password;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast(`已复制: ${password}`, 'success', 1500);
    });
}

// ============== Generate ==============

els.generateBtn.addEventListener('click', async () => {
    if (state.generating) return;

    const targetInfo = els.targetInfo.value.trim();
    if (!targetInfo) {
        showToast('请输入目标信息', 'warning');
        return;
    }

    state.generating = true;
    els.generateBtn.disabled = true;

    try {
        // Step 1: Load model if needed
        if (!state.modelLoaded) {
            showLoading('加载模型...');
            try {
                const result = await api.post('/api/load_model');
                state.modelLoaded = true;
                showToast(`模型已加载 (${result.parameters.toLocaleString()} 参数)`, 'success');
            } catch (e) {
                showToast(`模型加载失败: ${e.message}`, 'error');
                return;
            }
        }

        // Step 2: Generate
        showLoading('生成密码中...');

        const requestBody = {
            target_info: {
                raw_text: targetInfo,
                use_llm_extraction: state.llmConfigured && els.useLLM.checked,
                extraction_stages: parseInt(els.extractStages.value),
                parallel_extraction: els.parallelExtract.checked
            },
            generation: {
                method: els.genMethod.value,
                n_samples: parseInt(els.nSamples.value),
                temperature: parseFloat(els.temperature.value),
                temperature_schedule: els.tempSchedule.value,
                top_k: parseInt(els.topK.value),
                top_p: parseFloat(els.topP.value),
                beam_width: 5,
                diversity_penalty: 0.5,
                typical_mass: 0.9,
                contrastive_alpha: 0.5
            }
        };

        const result = await api.post('/api/generate', requestBody);

        // Render results
        renderFeatures(result.extracted_features);
        renderPasswords(result.passwords);

        // Update stats
        els.extractTime.textContent = result.extraction_time.toFixed(2) + 's';
        els.genTime.textContent = result.generation_time.toFixed(2) + 's';

        // Get cache stats
        try {
            const status = await api.get('/api/status');
            els.cacheHitRate.textContent = (status.cache_stats.hit_rate * 100).toFixed(1) + '%';
        } catch (e) {}

        showToast(`已生成 ${result.passwords.length} 个密码`, 'success');

    } catch (e) {
        showToast(`生成失败: ${e.message}`, 'error');
        console.error(e);
    } finally {
        state.generating = false;
        els.generateBtn.disabled = false;
        hideLoading();
    }
});

// ============== Utility ==============

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============== Init ==============

document.addEventListener('DOMContentLoaded', async () => {
    await checkStatus();

    // Show method description
    els.genMethod.dispatchEvent(new Event('change'));

    // Periodic status check
    setInterval(checkStatus, 30000);
});
