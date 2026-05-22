import { NextRequest, NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const OUTPUT_ROOT =
  process.env.SIMTOOLREAL_OUTPUT_ROOT ?? "/home/waybaba/code/simtoolreal/outputs";

function resolveOutputVideo(relativePath: string | null) {
  if (!relativePath) {
    throw new Error("missing path");
  }
  if (path.isAbsolute(relativePath) || relativePath.split(/[\\/]+/).includes("..")) {
    throw new Error("video path must stay inside the outputs directory");
  }
  if (!relativePath.endsWith(".mp4")) {
    throw new Error("only mp4 videos are supported");
  }

  const outputRoot = path.resolve(OUTPUT_ROOT);
  const videoPath = path.resolve(outputRoot, relativePath);
  if (!videoPath.startsWith(`${outputRoot}${path.sep}`)) {
    throw new Error("video path escaped the outputs directory");
  }
  return videoPath;
}

export async function GET(request: NextRequest) {
  try {
    const videoPath = resolveOutputVideo(request.nextUrl.searchParams.get("path"));
    const video = await readFile(videoPath);

    return new NextResponse(video, {
      headers: {
        "Content-Type": "video/mp4",
        "Content-Length": String(video.byteLength),
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to load video" },
      { status: 404 },
    );
  }
}
