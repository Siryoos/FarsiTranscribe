# Memory Optimization Guide for FarsiTranscribe

## Quick Memory Optimization

### 1. Use Memory-Optimized Preset
```bash
python main.py audio_file.m4a --quality memory-optimized
```

### 2. Use Smaller Models
- `tiny`: ~39MB RAM, fastest, lowest quality
- `base`: ~74MB RAM, fast, good quality
- `small`: ~244MB RAM, balanced
- `medium`: ~769MB RAM, high quality
- `large-v3`: ~1550MB RAM, highest quality

### 3. CPU-Only Installation (Saves GPU Memory)
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Minimal Dependencies Installation
```bash
pip install openai-whisper torch torchaudio pydub numpy tqdm
```

## Advanced Memory Management

### 1. Streaming Mode
The system automatically uses streaming mode for files >100MB or when `memory_efficient_mode=True`.

### 2. Chunk Size Optimization
- Smaller chunks = less memory per chunk
- Larger chunks = fewer processing steps
- Balance based on available RAM

### 3. Parallel Processing Control
- Reduce `num_workers` for lower memory usage
- Disable `use_parallel_audio_prep` for sequential processing

### 4. Preprocessing Control
- Disable `enable_preprocessing` for minimal memory usage
- Disable `enable_noise_reduction` to save memory
- Disable `enable_speech_enhancement` for faster processing

## Memory Usage by Configuration

| Configuration | Model Size | RAM Usage | Speed | Quality |
|---------------|------------|-----------|-------|---------|
| memory-optimized | small | ~300MB | Fast | Good |
| cpu-optimized | medium | ~800MB | Medium | High |
| balanced | large-v3 | ~1600MB | Slow | Best |
| high | large-v3 | ~2000MB | Slow | Best |

## Troubleshooting High Memory Usage

1. **Monitor Memory**: Use `--quality memory-optimized` with monitoring
2. **Close Other Apps**: Free up system RAM before transcription
3. **Use SSD**: Ensure sufficient disk space for temporary files
4. **Restart**: Restart the application between large files

## Environment Variables for Memory Control

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
```

## Performance vs Memory Trade-offs

- **Speed**: Use smaller models, disable preprocessing
- **Quality**: Use larger models, enable all preprocessing
- **Memory**: Use streaming mode, smaller chunks, fewer workers
- **Balance**: Use `--quality balanced` for optimal trade-offs
