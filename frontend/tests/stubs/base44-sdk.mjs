// Minimal stand-in for @base44/sdk as used by dapta-proxy; tests set globalThis.__base44User.
export function createClientFromRequest() {
  const entity = { filter: async () => [], create: async () => ({}), update: async () => ({}) };
  return {
    auth: { me: async () => globalThis.__base44User ?? null },
    asServiceRole: { entities: { DaptaSnapshot: entity } },
  };
}
