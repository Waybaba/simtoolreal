import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SimToolReal Research Studio",
  description: "Local canvas for SimToolReal IsaacGym and IsaacLab migration notes.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
