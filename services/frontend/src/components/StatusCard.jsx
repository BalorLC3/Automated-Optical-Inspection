export default function StatusCard({ qcStatus, message }) {
  const isPass = qcStatus === "PASS";

  return (
    <div
      className={`p-4 rounded-xl text-center font-semibold shadow-md ${
        isPass
          ? "bg-green-100 text-green-800"
          : "bg-red-100 text-red-800"
      }`}
    >
      <div className="text-xl mb-1">{qcStatus}</div>
      <div className="text-sm">{message}</div>
    </div>
  );
}
