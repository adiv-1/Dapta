import { createClient, getAccessToken } from "@base44/sdk";

export const base44 = createClient({
  appId: import.meta.env.VITE_BASE44_APP_ID || "",
  token: getAccessToken() || undefined,
  functionsVersion: import.meta.env.VITE_BASE44_FUNCTIONS_VERSION,
  appBaseUrl: import.meta.env.VITE_BASE44_APP_BASE_URL,
  serverUrl: "",
});
