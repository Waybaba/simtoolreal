import { NextRequest, NextResponse } from "next/server";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const OUTPUT_ROOT =
  process.env.SIMTOOLREAL_OUTPUT_ROOT ?? "/home/waybaba/code/simtoolreal/outputs";

function assertSafeOutputRelativePath(relativePath: string | null) {
  if (!relativePath) {
    throw new Error("missing dir");
  }
  if (path.isAbsolute(relativePath) || relativePath.split(/[\\/]+/).includes("..")) {
    throw new Error("directory must stay inside the outputs directory");
  }
  return relativePath;
}

function resolveOutputDir(relativePath: string | null) {
  const safePath = assertSafeOutputRelativePath(relativePath);
  const outputRoot = path.resolve(OUTPUT_ROOT);
  const dir = path.resolve(outputRoot, safePath);
  if (!dir.startsWith(`${outputRoot}${path.sep}`)) {
    throw new Error("directory escaped the outputs directory");
  }
  return dir;
}

function stepFromFileName(fileName: string) {
  return Number(fileName.match(/_video_(\d+)\.mp4$/)?.[1] ?? -1);
}

export async function GET(request: NextRequest) {
  try {
    const outputRoot = path.resolve(OUTPUT_ROOT);
    const dir = resolveOutputDir(request.nextUrl.searchParams.get("dir"));
    const entries = await readdir(dir, { withFileTypes: true });
    const videos = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".mp4"))
        .map(async (entry) => {
          const filePath = path.join(dir, entry.name);
          const fileStat = await stat(filePath);
          return {
            label: entry.name,
            path: path.relative(outputRoot, filePath),
            step: stepFromFileName(entry.name),
            sizeBytes: fileStat.size,
            modifiedMs: fileStat.mtimeMs,
          };
        }),
    );

    videos.sort((left, right) => {
      if (left.step !== right.step) {
        return left.step - right.step;
      }
      return left.label.localeCompare(right.label);
    });

    return NextResponse.json({ videos });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to list videos" },
      { status: 404 },
    );
  }
}
