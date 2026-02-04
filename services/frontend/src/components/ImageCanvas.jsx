import { useRef, useEffect } from "react";

export default function ImageCanvas({ imageSrc, detections }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageSrc) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Draw detections with industrial style
      detections.forEach(det => {
        const { x, y, w, h, label, confidence } = det;

        // Draw bounding box
        ctx.strokeStyle = "#ff3d00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        const text = `${label} ${(confidence * 100).toFixed(1)}%`;
        ctx.font = "bold 14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const textHeight = 20;

        ctx.fillStyle = "#ff3d00";
        ctx.fillRect(x, y - textHeight - 4, textWidth + 12, textHeight + 4);

        // Draw label text
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, x + 6, y - 8);
      });
    };

    img.src = imageSrc;
  }, [imageSrc, detections]);

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        className="canvas-display"
      />
    </div>
  );
}