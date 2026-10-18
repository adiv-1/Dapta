// Stand-in for Base44 function secrets; tests set globalThis.__base44Secrets.
export const secrets = { get: (name) => (globalThis.__base44Secrets ?? {})[name] };
