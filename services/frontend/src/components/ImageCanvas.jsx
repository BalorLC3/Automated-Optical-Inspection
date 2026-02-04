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

      // Draw detections (YOLO-style boxes)
      detections.forEach(det => {
        const { x, y, w, h, label, confidence } = det;

        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "red";
        ctx.font = "14px sans-serif";
        ctx.fillText(
          `${label} (${(confidence * 100).toFixed(1)}%)`,
          x,
          y - 5
        );
      });
    };

    img.src = imageSrc;
  }, [imageSrc, detections]);

  return (
    <canvas
      ref={canvasRef}
      className="max-w-full h-auto border rounded-lg"
    />
  );
}
