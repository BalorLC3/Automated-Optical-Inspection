export default function StatusCard({ qcStatus, message }) {
  const isPass = qcStatus?.toLowerCase() === "pass";

  return (
    <div className="status-card">
      {/* Status Bar */}
      <div className={`status-bar ${isPass ? 'status-bar-pass' : 'status-bar-fail'}`} />
      
      {/* Card Content */}
      <div className="status-content">
        <div className={`status-label ${isPass ? 'status-ok' : 'status-error'}`}>
          {qcStatus}
        </div>
        <div className="status-message">
          {message}
        </div>
      </div>
    </div>
  );
}