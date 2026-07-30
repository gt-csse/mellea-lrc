import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "mellea-LRC",
  description: "Read-only visualization of mellea-lrc validation artifacts"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
