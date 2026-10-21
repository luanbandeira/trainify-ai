const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../app/static/index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function ambiente() {
  const elementos = new Map();
  const armazenamento = new Map();
  function elemento() {
    return {
      value: "", disabled: false, hidden: false, children: [], handlers: {},
      classList: { add() {}, remove() {} },
      addEventListener(evento, callback) { this.handlers[evento] = callback; },
      append(...itens) { for (const item of itens) { item.parent = this; this.children.push(item); } },
      remove() { this.parent.children = this.parent.children.filter((item) => item !== this); },
      querySelectorAll() { return this.children; },
      focus() {}, setAttribute() {},
    };
  }
  const get = (id) => {
    if (!elementos.has(id)) elementos.set(id, elemento());
    return elementos.get(id);
  };
  const contexto = vm.createContext({
    document: { getElementById: get, querySelectorAll: () => [], createElement: elemento },
    localStorage: {
      getItem: (key) => armazenamento.get(key),
      setItem: (key, value) => armazenamento.set(key, value),
      removeItem: (key) => armazenamento.delete(key),
    },
    fetch: () => new Promise(() => {}),
    AbortController, TextDecoderStream, setTimeout, clearTimeout,
  });
  vm.runInContext(script, contexto);
  return {
    contexto, get, armazenamento,
    perguntar: (texto) => contexto.ask(texto),
    limpar: () => get("clear").handlers.click(),
    historico: () => vm.runInContext("JSON.stringify(history)", contexto),
  };
}

function controlarRespostas(app) {
  const pedidos = [];
  app.contexto.chatEmStream = (body, token, controle) => new Promise((resolve, reject) => {
    pedidos.push({ body, token, controle, resolve, reject });
  });
  return pedidos;
}

test("limpar cancela e ignora tokens e conclusão atrasados, inclusive no armazenamento", async () => {
  const app = ambiente();
  const pedidos = controlarRespostas(app);
  const pendente = app.perguntar("Pergunta antiga");
  pedidos[0].token("Resposta parcial");
  app.limpar();
  assert.equal(pedidos[0].controle.signal.aborted, true);
  pedidos[0].token("Token atrasado");
  pedidos[0].resolve({ answer: "Resposta antiga", sources: [] });
  await pendente;
  assert.equal(app.get("messages").children.length, 0);
  assert.equal(app.historico(), "[]");
  assert.equal(app.armazenamento.size, 0);
  assert.equal(app.get("empty").hidden, false);
  assert.equal(app.get("send").disabled, false);
});

test("falha antiga não altera os controles de uma nova pergunta", async () => {
  const app = ambiente();
  const pedidos = controlarRespostas(app);
  const antiga = app.perguntar("Antiga");
  app.limpar();
  const nova = app.perguntar("Nova");
  pedidos[0].reject(new Error("Falha antiga"));
  await antiga;
  assert.equal(app.get("send").disabled, true);
  assert.equal(app.get("messages").children.length, 2);
  pedidos[1].resolve({ answer: "Resposta nova", sources: [] });
  await nova;
  assert.equal(app.get("send").disabled, false);
  assert.deepEqual(JSON.parse(app.historico()).map((m) => m.content), ["Nova", "Resposta nova"]);
  assert.equal(app.armazenamento.size, 1);
});

test("envios repetidos não iniciam outra execução e limpar pode ser repetido", async () => {
  const app = ambiente();
  const pedidos = controlarRespostas(app);
  const pendente = app.perguntar("Primeira");
  await app.perguntar("Segunda");
  assert.equal(pedidos.length, 1);
  app.limpar();
  app.limpar();
  pedidos[0].resolve({ answer: "Antiga", sources: [] });
  await pendente;
  assert.equal(app.get("messages").children.length, 0);
});

test("limpar aborta a requisição do chat sem mostrar erro de timeout", async () => {
  const app = ambiente();
  let sinal;
  app.contexto.fetch = (_, options) => new Promise((resolve, reject) => {
    sinal = options.signal;
    sinal.addEventListener("abort", () => reject(Object.assign(new Error("Cancelado"), { name: "AbortError" })));
  });
  const pendente = app.perguntar("Pergunta");
  app.limpar();
  await pendente;
  assert.equal(sinal.aborted, true);
  assert.equal(app.get("messages").children.length, 0);
  assert.equal(app.armazenamento.size, 0);
});


test("timeout da conversa ativa continua mostrando o erro e libera novo envio", async () => {
  const app = ambiente();
  let expirar;
  app.contexto.setTimeout = (callback) => { expirar = callback; return 1; };
  app.contexto.clearTimeout = () => {};
  app.contexto.fetch = (_, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener("abort", () => reject(Object.assign(new Error("Timeout"), { name: "AbortError" })));
  });
  const pendente = app.perguntar("Pergunta");
  expirar();
  await pendente;
  const erro = app.get("messages").children.find((item) => item.className === "msg error");
  assert.match(erro.textContent, /demorou demais/);
  assert.equal(app.get("send").disabled, false);
  assert.equal(app.historico(), "[]");
});


test("resposta condensada mostra aviso e salva a indicação no histórico", async () => {
  const app = ambiente();
  const pedidos = controlarRespostas(app);
  const pendente = app.perguntar("Resuma a aula");
  pedidos[0].resolve({ answer: "Resumo", sources: [], context_condensed: true });
  await pendente;
  const resposta = app.get("messages").children[1];
  assert.ok(resposta.children.some((item) => item.textContent?.includes("versão reduzida")));
  const salvo = JSON.parse([...app.armazenamento.values()][0]);
  assert.equal(salvo.mensagens[1].context_condensed, true);
  const restaurada = app.contexto.document.createElement("div");
  app.contexto.mostrarResposta(restaurada, salvo.mensagens[1]);
  assert.ok(restaurada.children.some((item) => item.textContent?.includes("versão reduzida")));
});

test("conversas antigas sem indicação continuam legíveis e sem aviso", () => {
  const app = ambiente();
  const resposta = app.contexto.document.createElement("div");
  app.contexto.mostrarResposta(resposta, { text: "Resposta antiga", sources: [] });
  assert.equal(resposta.children.length, 0);
});


for (const nivel of ["", "BASIC", "INTERMEDIATE", "ADVANCED"]) {
  test(`formulário envia dificuldade ${nivel || "padrão"} sem alterar o escopo`, async () => {
    const app = ambiente();
    app.get("quizScope").value = "lesson:nr-06";
    app.get("quizN").value = "3";
    app.get("quizOpts").value = "4";
    app.get("quizDifficulty").value = nivel;
    app.get("quizOut").replaceChildren = () => {};
    let enviado;
    app.contexto.post = async (url, body) => {
      enviado = { url, body };
      return { title: "Quiz", passingScore: 70, sources: [], questions: [] };
    };
    await app.get("quizForm").handlers.submit({ preventDefault() {} });
    assert.equal(enviado.url, "/demo/quiz");
    assert.equal(enviado.body.difficulty, nivel || null);
    assert.equal(enviado.body.lesson_id, "nr-06");
    assert.equal(enviado.body.module_id, null);
    assert.equal(enviado.body.num_questions, 3);
    assert.equal(app.get("quizBtn").disabled, false);
  });
}
