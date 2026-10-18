// Stand-in for src/base44Client.ts: records dapta-proxy invocations; tests set the reply.
export const invocations = [];
export const base44 = {
  functions: {
    async invoke(name, payload) {
      invocations.push({ name, payload });
      const reply = globalThis.__base44Reply;
      if (!reply) return { data: [] };
      return reply(name, payload);
    },
  },
};
