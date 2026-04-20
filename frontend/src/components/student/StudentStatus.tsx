import React from 'react';
import './StudentStatus.css';

interface StudentStatusProps {
    studentName: string;
    studentId: string;
    registered: boolean;
    registeredFacesCount: number | 0;
}

const StudentStatus: React.FC<StudentStatusProps> = ({
  studentName,
  studentId,
  registered,
  registeredFacesCount
}) => (
    <div className="student-status">
        <div className="student-status__top" />
        <div className="student-status__content">
            <h2 className="student-status__name">{studentName}</h2>
            <p className="student-status__id">
                <span className="student-status__label">ID:</span> {studentId}
            </p>
            <div className={`student-status__badge ${registered ? 'student-status__badge--registered' : 'student-status__badge--unregistered'}`}>
                {registered ? `✓ ${registeredFacesCount} face${registeredFacesCount !== 1 ? 's' : ''} registered` : '○ Not registered'}
            </div>
        </div>
    </div>
);

export default StudentStatus;