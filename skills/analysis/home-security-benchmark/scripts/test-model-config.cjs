#!/usr/bin/env node
/**
 * Unit tests for MODEL_FAMILIES / getModelApiParams logic.
 *
 * Tests the model-family detection and per-request param injection
 * without needing a running LLM server.
 *
 * Usage:
 *   node scripts/test-model-config.cjs
 */

// ── Inline the config under test ─────────────────────────────────────────────
// (Kept in sync with run-benchmark.cjs MODEL_FAMILIES section)

const MODEL_FAMILIES = [
    {
        name: 'Mistral',
        match: (m) => m.includes('mistral') || m.includes('magistral') || m.includes('mixtral'),
        apiParams: { reasoning_effort: 'none' },
        serverFlags: '--reasoning-budget 0',
    },
    // Qwen3.5: no extra per-request params needed (handled by prompt + abort logic)
];

function getModelApiParams(modelName) {
    if (!modelName) return {};
    const lower = modelName.toLowerCase();
    for (const family of MODEL_FAMILIES) {
        if (family.match(lower)) return family.apiParams || {};
    }
    return {};
}

// ── Mirror the server-manager detection ──────────────────────────────────────
function getServerFlags(modelFilePath) {
    const lower = modelFilePath.toLowerCase();
    const isMistralFamily = lower.includes('mistral') ||
                            lower.includes('magistral') ||
                            lower.includes('mixtral');
    return isMistralFamily
        ? { flag: '--reasoning-budget', value: '0' }
        : { flag: '--chat-template-kwargs', value: '{"enable_thinking":false}' };
}

// ── Test harness ─────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`  ✅ ${name}`);
        passed++;
    } catch (err) {
        console.log(`  ❌ ${name}: ${err.message}`);
        failed++;
    }
}

function assert(condition, msg) {
    if (!condition) throw new Error(msg || 'Assertion failed');
}

function assertDeepEqual(a, b, msg) {
    const as = JSON.stringify(a), bs = JSON.stringify(b);
    if (as !== bs) throw new Error(`${msg || 'Not equal'}: got ${as}, expected ${bs}`);
}

// ── Tests ────────────────────────────────────────────────────────────────────

console.log('\n=== MODEL_FAMILIES / getModelApiParams ===\n');

// ── Mistral detection ─────────────────────────────────────────────────────────
test('Mistral-Small-4-119B GGUF filename → reasoning_effort:none', () => {
    const p = getModelApiParams('Mistral-Small-4-119B-2603-UD-IQ1_M.gguf');
    assertDeepEqual(p, { reasoning_effort: 'none' });
});

test('Mistral-Small-4 Q2_K_XL variant → reasoning_effort:none', () => {
    const p = getModelApiParams('Mistral-Small-4-119B-2603-UD-Q2_K_XL.gguf');
    assertDeepEqual(p, { reasoning_effort: 'none' });
});

test('Magistral model → reasoning_effort:none', () => {
    const p = getModelApiParams('magistral-medium-2506.gguf');
    assertDeepEqual(p, { reasoning_effort: 'none' });
});

test('Mixtral-8x7B → reasoning_effort:none', () => {
    const p = getModelApiParams('Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf');
    assertDeepEqual(p, { reasoning_effort: 'none' });
});

test('Mistral cloud API model ID → reasoning_effort:none', () => {
    const p = getModelApiParams('mistral-small-latest');
    assertDeepEqual(p, { reasoning_effort: 'none' });
});

// ── Non-Mistral: should get no extra params ───────────────────────────────────
test('Qwen3.5-9B → no extra params (handled by prompt)', () => {
    const p = getModelApiParams('Qwen3.5-9B-Q4_K_M.gguf');
    assertDeepEqual(p, {});
});

test('Qwen3.5-27B → no extra params', () => {
    const p = getModelApiParams('Qwen3.5-27B-UD-Q8_K_XL.gguf');
    assertDeepEqual(p, {});
});

test('NVIDIA Nemotron-30B → no extra params', () => {
    const p = getModelApiParams('NVIDIA-Nemotron-3-Nano-30B-A3B-Q8_0.gguf');
    assertDeepEqual(p, {});
});

test('LFM2-24B → no extra params', () => {
    const p = getModelApiParams('LFM2-24B-A2B-Q8_0.gguf');
    assertDeepEqual(p, {});
});

test('GPT-5.4 → no extra params', () => {
    const p = getModelApiParams('gpt-5.4-2026-03-05');
    assertDeepEqual(p, {});
});

test('Empty model name → no extra params', () => {
    const p = getModelApiParams('');
    assertDeepEqual(p, {});
});

test('Undefined model name → no extra params', () => {
    const p = getModelApiParams(undefined);
    assertDeepEqual(p, {});
});

// ── Server-manager flags (mirrors llm-server-manager.cjs logic) ───────────────
console.log('\n=== Server-manager startup flags ===\n');

test('Mistral GGUF path → --reasoning-budget 0', () => {
    const f = getServerFlags('/Users/simba/.aegis-ai/models/Mistral-Small-4-119B-2603-UD-IQ1_M.gguf');
    assert(f.flag === '--reasoning-budget' && f.value === '0',
        `Expected --reasoning-budget 0, got ${f.flag} ${f.value}`);
});

test('Magistral path → --reasoning-budget 0', () => {
    const f = getServerFlags('/models/magistral-medium.gguf');
    assert(f.flag === '--reasoning-budget' && f.value === '0');
});

test('Qwen path → --chat-template-kwargs enable_thinking:false', () => {
    const f = getServerFlags('/models/Qwen3.5-9B-Q4_K_M.gguf');
    assert(f.flag === '--chat-template-kwargs');
    assert(f.value.includes('enable_thinking'));
    assert(f.value.includes('false'));
});

test('Nemotron path → --chat-template-kwargs enable_thinking:false', () => {
    const f = getServerFlags('/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Q8_0.gguf');
    assert(f.flag === '--chat-template-kwargs');
});

test('LFM2 path → --chat-template-kwargs enable_thinking:false', () => {
    const f = getServerFlags('/models/LFM2-24B-A2B-Q8_0.gguf');
    assert(f.flag === '--chat-template-kwargs');
});

// ── Summary ──────────────────────────────────────────────────────────────────

console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
