# Chunk Analysis Guide for FarsiTranscribe

## 📊 Understanding Chunks

Chunks are the way FarsiTranscribe breaks down large audio files into smaller, manageable pieces for processing. This guide explains how to understand and analyze chunk information.

## 🔍 How to Get Chunk Information

### 1. **During Transcription**
When you run a transcription, the system now shows detailed chunk information:

```bash
python main.py audio.m4a --quality memory-optimized
```

**Output includes:**
```
📊 Audio Analysis:
   Duration: 7271.6 seconds
   Total Chunks: 731
   Chunk Duration: 10000ms (10.0s)
   Overlap: 50ms (0.1s)
   Effective Chunk Duration: 9950ms (9.9s)
```

### 2. **Pre-Analysis Script**
Use the dedicated analysis script to see chunk details before processing:

```bash
# Analyze with memory-optimized configuration
python scripts/analyze_chunks.py audio.m4a

# Compare all configurations
python scripts/analyze_chunks.py audio.m4a --compare

# Analyze with specific configuration
python scripts/analyze_chunks.py audio.m4a --config cpu-optimized
```

## 📋 Chunk Information Explained

### **Basic Chunk Parameters**

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Total Chunks** | Number of pieces the audio will be split into | 731 |
| **Chunk Duration** | Length of each chunk in milliseconds | 10000ms (10s) |
| **Overlap** | Overlap between consecutive chunks | 50ms (0.05s) |
| **Effective Duration** | Actual new content per chunk | 9950ms (9.95s) |

### **Chunk Calculation Formula**

```
Effective Chunk Duration = Chunk Duration - Overlap
Total Chunks = ceil((Audio Duration - Overlap) / Effective Chunk Duration)
```

**Example:**
- Audio: 7271.6 seconds (7,271,573ms)
- Chunk Duration: 10,000ms
- Overlap: 50ms
- Effective Duration: 9,950ms
- Total Chunks: ceil((7,271,573 - 50) / 9,950) = 731

## 🎯 Configuration Impact on Chunks

### **Memory-Optimized Configuration**
```bash
python scripts/analyze_chunks.py audio.m4a --config memory-optimized
```
- **Model**: small
- **Chunk Duration**: 10 seconds
- **Overlap**: 50ms
- **Result**: More chunks, lower memory usage
- **Use Case**: Low RAM systems, large files

### **CPU-Optimized Configuration**
```bash
python scripts/analyze_chunks.py audio.m4a --config cpu-optimized
```
- **Model**: medium
- **Chunk Duration**: 25 seconds
- **Overlap**: 300ms
- **Result**: Fewer chunks, balanced performance
- **Use Case**: Medium RAM systems

### **Balanced Configuration**
```bash
python scripts/analyze_chunks.py audio.m4a --config balanced
```
- **Model**: large
- **Chunk Duration**: 20 seconds
- **Overlap**: 500ms
- **Result**: Moderate chunks, high quality
- **Use Case**: High RAM systems

### **High Quality Configuration**
```bash
python scripts/analyze_chunks.py audio.m4a --config high
```
- **Model**: large-v3
- **Chunk Duration**: 15 seconds
- **Overlap**: 300ms
- **Result**: More chunks, highest quality
- **Use Case**: Maximum quality, sufficient RAM

## 📊 Chunk Comparison Example

For a 2-hour audio file (7,271 seconds):

| Configuration | Model | Chunks | Chunk Duration | Overlap | Memory Usage |
|---------------|-------|--------|----------------|---------|--------------|
| memory-optimized | small | 731 | 10s | 50ms | ~300MB |
| cpu-optimized | medium | 295 | 25s | 300ms | ~800MB |
| balanced | large | 373 | 20s | 500ms | ~1600MB |
| high | large-v3 | 495 | 15s | 300ms | ~2000MB |

## 💾 Memory Impact

### **Memory Per Chunk Calculation**
```
Memory per chunk = (Chunk Duration × Sample Rate × Channels × 2) / (1024 × 1024)
```

**Example:**
- Chunk Duration: 10,000ms
- Sample Rate: 48,000 Hz
- Channels: 2
- Memory per chunk: (10,000 × 48,000 × 2 × 2) / (1024 × 1024) ≈ 1,831 MB

### **Peak Memory Usage**
- **Streaming Mode**: Processes 1-3 chunks at a time
- **Batch Mode**: Processes multiple chunks simultaneously
- **Memory Efficient**: Uses streaming for large files

## ⏱️ Processing Time Estimation

### **Time Per Chunk by Model**
| Model | Estimated Time per Chunk |
|-------|-------------------------|
| tiny | 2 seconds |
| base | 4 seconds |
| small | 8 seconds |
| medium | 15 seconds |
| large-v3 | 30 seconds |

### **Total Processing Time**
```
Total Time = Number of Chunks × Time per Chunk
```

**Example for 731 chunks with small model:**
- Total Time = 731 × 8 seconds = 5,848 seconds ≈ 97 minutes

## 🔧 Advanced Chunk Analysis

### **Chunk Details**
The analysis script shows individual chunk information:

```
🔢 Chunk Details:
   Chunk  1:      0ms -  10000ms ( 10000ms)
   Chunk  2:   9950ms -  19950ms ( 10000ms)
   Chunk  3:  19900ms -  29900ms ( 10000ms)
   ...
   Chunk 731: 7263500ms - 7271573ms (  8073ms)
```

### **Chunk Timing**
- **Start Time**: When each chunk begins
- **End Time**: When each chunk ends
- **Duration**: Actual length of the chunk
- **Overlap**: Shared content with adjacent chunks

## 🚀 Optimization Strategies

### **For Large Files (>100 chunks)**
1. **Use memory-optimized configuration**
2. **Enable streaming mode**
3. **Monitor memory usage**
4. **Consider smaller models**

### **For Small Files (<50 chunks)**
1. **Use balanced or high quality**
2. **Enable full preprocessing**
3. **Use larger models for better quality**

### **For Medium Files (50-100 chunks)**
1. **Use cpu-optimized configuration**
2. **Balance quality and speed**
3. **Monitor system resources**

## 📈 Progress Monitoring

### **During Transcription**
The system shows real-time progress:

```
📊 Progress: 45/731 chunks (6.2%)
📊 Progress: 90/731 chunks (12.3%)
📊 Progress: 135/731 chunks (18.5%)
```

### **Progress Calculation**
```
Progress Percentage = (Current Chunk / Total Chunks) × 100
```

## 🛠️ Troubleshooting

### **High Memory Usage**
- **Problem**: Too many chunks in memory
- **Solution**: Use memory-optimized configuration
- **Alternative**: Reduce chunk duration

### **Slow Processing**
- **Problem**: Too many small chunks
- **Solution**: Increase chunk duration
- **Alternative**: Use faster model

### **Poor Quality**
- **Problem**: Chunks too large
- **Solution**: Reduce chunk duration
- **Alternative**: Use larger model

## 📚 Best Practices

### **1. Pre-Analyze Your Files**
```bash
python scripts/analyze_chunks.py your_audio.m4a --compare
```

### **2. Choose Configuration Based on File Size**
- **Small files (< 10 minutes)**: balanced or high
- **Medium files (10-60 minutes)**: cpu-optimized
- **Large files (> 60 minutes)**: memory-optimized

### **3. Monitor System Resources**
- Check available RAM before processing
- Use streaming mode for large files
- Monitor progress during transcription

### **4. Optimize for Your Use Case**
- **Speed**: Use smaller models, larger chunks
- **Quality**: Use larger models, smaller chunks
- **Memory**: Use streaming mode, memory-optimized config

## 🎯 Quick Reference

### **Common Commands**
```bash
# Analyze chunks before processing
python scripts/analyze_chunks.py audio.m4a

# Compare all configurations
python scripts/analyze_chunks.py audio.m4a --compare

# Process with chunk information
python main.py audio.m4a --quality memory-optimized
```

### **Configuration Selection**
| File Size | Recommended Config | Expected Chunks |
|-----------|-------------------|-----------------|
| < 10 min | balanced | 20-50 |
| 10-60 min | cpu-optimized | 50-200 |
| > 60 min | memory-optimized | 200+ |

### **Memory Requirements**
| Configuration | Peak Memory | Suitable For |
|---------------|-------------|--------------|
| memory-optimized | ~300MB | Low RAM systems |
| cpu-optimized | ~800MB | Medium RAM systems |
| balanced | ~1600MB | High RAM systems |
| high | ~2000MB | Maximum quality |

---

**💡 Pro Tip**: Always run the chunk analysis script before processing large files to understand the memory and time requirements! 