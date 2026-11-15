/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string
  readonly VITE_NO_SERVER?: string
  // folder in public/ with data downloaded in advance (e.g. "haifa"): the site then needs no server
  readonly VITE_STATIC_DATA?: string
}
