import React from 'react';
import './SaveModal.css';

interface SaveModalProps {
  isOpen: boolean;
  message: string;
}

const SaveModal: React.FC<SaveModalProps> = ({ isOpen, message }) => {
  if (!isOpen) return null;

  return (
    <div className="save-modal-overlay">
      <div className="save-modal-content">
        <div className="spinner"></div>
        <p className="save-modal-message">{message}</p>
      </div>
    </div>
  );
};

export default SaveModal;
