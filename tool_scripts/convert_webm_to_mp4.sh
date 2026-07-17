#!/usr/bin/env bash
# Convert Ubuntu screen recordings (.webm) to broadly compatible MP4 files.
# Requires: ffmpeg (sudo apt install ffmpeg)

set -Eeuo pipefail

usage() {
    cat <<'EOF'
用法:
  convert_webm_to_mp4.sh <输入.webm> [输出.mp4] [选项]
  convert_webm_to_mp4.sh <包含 webm 的目录> [输出目录] [选项]

示例:
  # 单个文件：默认在原文件旁生成 recording.mp4
  ./scripts/convert_webm_to_mp4.sh ~/Videos/recording.webm

  # 指定单个文件的输出路径
  ./scripts/convert_webm_to_mp4.sh ~/Videos/recording.webm ~/Videos/demo.mp4

  # 批量转换目录内（含子目录）的所有 .webm，并保留目录结构
  ./scripts/convert_webm_to_mp4.sh ~/Videos/Recordings ~/Videos/MP4

选项:
  -f, --force             覆盖已有 MP4。
  --fps <帧率>            输出帧率，默认 30。可设为 60。
  --preset <预设>         libx264 预设，默认 veryfast（更快）。
  --crf <0-51>            画质参数，默认 22；数字越小画质越高、速度越慢。

默认不覆盖已有 MP4；传入 --force 才会覆盖。
转换中会每秒显示进度、已用时间和预计剩余时间。
EOF
}

force=0
target_fps=30
preset=veryfast
crf=22
args=()
while (( $# > 0 )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -f|--force)
            force=1
            shift
            ;;
        --fps|--preset|--crf)
            if (( $# < 2 )); then
                echo "错误：$1 需要一个参数。" >&2
                exit 2
            fi
            case "$1" in
                --fps) target_fps=$2 ;;
                --preset) preset=$2 ;;
                --crf) crf=$2 ;;
            esac
            shift 2
            ;;
        --)
            shift
            args+=("$@")
            break
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

if (( ${#args[@]} < 1 || ${#args[@]} > 2 )); then
    usage >&2
    exit 2
fi

if [[ ! "$target_fps" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]]; then
    echo "错误：--fps 必须是正数，例如 --fps 30。" >&2
    exit 2
fi

if [[ ! "$crf" =~ ^[0-9]+$ ]] || (( crf < 0 || crf > 51 )); then
    echo "错误：--crf 必须是 0 到 51 的整数。" >&2
    exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "错误：未找到 ffmpeg。请先安装：sudo apt update && sudo apt install ffmpeg" >&2
    exit 127
fi

input=${args[0]}
requested_output=${args[1]:-}

if [[ ! -e "$input" ]]; then
    echo "错误：输入路径不存在：$input" >&2
    exit 1
fi

format_duration() {
    local total_seconds=$1
    printf '%02d:%02d:%02d' \
        "$((total_seconds / 3600))" \
        "$(((total_seconds % 3600) / 60))" \
        "$((total_seconds % 60))"
}

convert_one() {
    local source=$1
    local destination=$2
    local duration_seconds duration_us
    local start_time
    local -a overwrite_option=(-n)

    if [[ -e "$destination" && $force -eq 0 ]]; then
        echo "跳过（已存在）：$destination"
        return 0
    fi

    if (( force == 1 )); then
        overwrite_option=(-y)
    fi

    mkdir -p "$(dirname "$destination")"
    echo "转换：$source"

    duration_seconds=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$source" 2>/dev/null || true)
    if [[ "$duration_seconds" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        duration_us=$(LC_ALL=C awk -v seconds="$duration_seconds" 'BEGIN { printf "%.0f", seconds * 1000000 }')
        echo "输出：H.264 / ${target_fps} fps / preset=${preset} / crf=${crf}"
        echo "进度：  0% | 已用：00:00:00 | 预计剩余：计算中…"
    else
        duration_us=0
        echo "输出：H.264 / ${target_fps} fps / preset=${preset} / crf=${crf}"
        echo "进度：时长未知，正在转换…"
    fi

    start_time=$(date +%s)
    if ! ffmpeg -hide_banner -loglevel error "${overwrite_option[@]}" \
        -i "$source" \
        -map 0:v:0 -map 0:a? \
        -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' \
        -r "$target_fps" \
        -c:v libx264 -preset "$preset" -crf "$crf" -pix_fmt yuv420p \
        -c:a aac \
        -movflags +faststart \
        -progress pipe:1 -nostats \
        "$destination" | {
            local processed_us=0
            local last_update=-1
            local key value now elapsed percent remaining eta speed_x100

            while IFS='=' read -r key value; do
                [[ "$key" == out_time_us ]] || continue
                processed_us=${value:-0}
                (( duration_us > 0 && processed_us > duration_us )) && processed_us=$duration_us
                now=$(date +%s)
                (( now == last_update )) && continue
                last_update=$now
                elapsed=$((now - start_time))

                if (( duration_us > 0 && processed_us > 0 && elapsed > 0 )); then
                    percent=$((processed_us * 100 / duration_us))
                    remaining=$((duration_us - processed_us))
                    eta=$((elapsed * remaining / processed_us / 1000000))
                    speed_x100=$((processed_us * 100 / (elapsed * 1000000)))
                    printf '\r进度：%3d%% | 已用：%s | 预计剩余：%s | 速度：%d.%02dx' \
                        "$percent" "$(format_duration "$elapsed")" "$(format_duration "$eta")" \
                        "$((speed_x100 / 100))" "$((speed_x100 % 100))"
                fi
            done
        }; then
        printf '\n转换失败：%s\n' "$source" >&2
        return 1
    fi
    printf '\r进度：100%% | 已用：%s | 预计剩余：00:00:00                    \n' \
        "$(format_duration "$(( $(date +%s) - start_time ))")"
    echo "完成：$destination"
}

if [[ -f "$input" ]]; then
    if [[ "${input,,}" != *.webm ]]; then
        echo "错误：输入文件必须是 .webm：$input" >&2
        exit 2
    fi

    if [[ -n "$requested_output" ]]; then
        output=$requested_output
    else
        output="${input%.*}.mp4"
    fi
    if [[ "${output,,}" != *.mp4 ]]; then
        echo "错误：单个文件的输出路径必须以 .mp4 结尾：$output" >&2
        exit 2
    fi
    convert_one "$input" "$output"
    exit 0
fi

if [[ -n "$requested_output" ]]; then
    output_dir=$requested_output
else
    output_dir="$input/mp4"
fi

converted=0
while IFS= read -r -d '' source; do
    relative_path=${source#"${input%/}"/}
    output="$output_dir/${relative_path%.*}.mp4"
    convert_one "$source" "$output"
    ((converted += 1))
done < <(find "$input" -type f -iname '*.webm' -print0)

if (( converted == 0 )); then
    echo "未找到 .webm 文件：$input" >&2
    exit 1
fi

echo "批量转换完成，共处理 $converted 个文件。"
