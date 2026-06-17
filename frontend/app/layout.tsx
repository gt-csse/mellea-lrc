import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mellea LRC Review",
  description: "Citation extraction and validation review workspace"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
