import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        medicalBlue: "#1e40af",
        medicalCyan: "#0ea5e9"
      },
      fontFamily: {
        sans: ["DM Sans", "sans-serif"],
        mono: ["DM Mono", "monospace"]
      }
    }
  },
  plugins: []
} satisfies Config;
