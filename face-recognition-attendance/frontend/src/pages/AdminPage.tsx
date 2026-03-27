import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  getAttendanceLogs,
  exportAttendanceCsv,
  type AttendanceLogEntry,
} from '../services/adminService';
import './AdminPage.css';

const getTodayStr = (): string => {
  const d = new Date();
  return d.toISOString().slice(0, 10); // YYYY-MM-DD
};

const AdminPage = () => {
  const navigate = useNavigate();

  // Date filters — default to today
  const [startDate, setStartDate] = useState(getTodayStr());
  const [endDate, setEndDate] = useState(getTodayStr());

  // Data
  const [logs, setLogs] = useState<AttendanceLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const fetchLogs = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await getAttendanceLogs(startDate, endDate);
      setLogs(data);
    } catch (err) {
      console.error('Failed to fetch logs:', err);
      setLogs([]);
    } finally {
      setIsLoading(false);
    }
  }, [startDate, endDate]);

  // Fetch on mount
  useEffect(() => {
    fetchLogs();
  }, []);

  const handleFilter = () => {
    fetchLogs();
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      await exportAttendanceCsv(startDate, endDate);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setIsExporting(false);
    }
  };

  // Summary stats
  const totalLogs = logs.length;
  const presentCount = logs.filter((l) => l.status === 'present').length;
  const absentCount = logs.filter((l) => l.status === 'absent').length;

  return (
    <div className="admin-page">
      <h1 className="admin-page__title">Admin Panel</h1>

      <main className="admin-page__main">
        {/* Summary Cards */}
        <div className="admin-page__summary">
          <div className="admin-page__summary-card">
            <p className="admin-page__summary-value admin-page__summary-value--total">
              {totalLogs}
            </p>
            <p className="admin-page__summary-label">Total Records</p>
          </div>
          <div className="admin-page__summary-card">
            <p className="admin-page__summary-value admin-page__summary-value--present">
              {presentCount}
            </p>
            <p className="admin-page__summary-label">Present</p>
          </div>
          <div className="admin-page__summary-card">
            <p className="admin-page__summary-value admin-page__summary-value--absent">
              {absentCount}
            </p>
            <p className="admin-page__summary-label">Absent</p>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="admin-page__filters">
          <div className="admin-page__filter-group">
            <label>From:</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="admin-page__filter-group">
            <label>To:</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <button className="admin-page__filter-btn" onClick={handleFilter}>
            Apply Filter
          </button>
          <button
            className="admin-page__export-btn"
            onClick={handleExport}
            disabled={isExporting || logs.length === 0}
          >
            {isExporting ? 'Exporting...' : 'Export CSV'}
          </button>
        </div>

        {/* Table */}
        <div className="admin-page__table-wrapper">
          {isLoading && (
            <div className="admin-page__loading">
              <div className="admin-page__spinner" />
              <p>Loading attendance logs...</p>
            </div>
          )}

          {!isLoading && logs.length === 0 && (
            <div className="admin-page__empty">
              <p>No attendance records found for the selected period.</p>
            </div>
          )}

          {!isLoading && logs.length > 0 && (
            <table className="admin-page__table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Student ID</th>
                  <th>Student Name</th>
                  <th>Status</th>
                  <th>Similarity</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    <td>{log.student_id}</td>
                    <td>{log.student_name}</td>
                    <td>
                      <span
                        className={
                          log.status === 'present'
                            ? 'admin-page__status--present'
                            : 'admin-page__status--absent'
                        }
                      >
                        {log.status}
                      </span>
                    </td>
                    <td>{(log.similarity * 100).toFixed(1)}%</td>
                    <td>{new Date(log.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>

      <button className="admin-page__back-btn" onClick={() => navigate('/')}>
        ← Back to Dashboard
      </button>
    </div>
  );
};

export default AdminPage;