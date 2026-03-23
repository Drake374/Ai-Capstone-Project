import React from "react";
import { useNavigate } from "react-router-dom";
import { logoutUser } from "../services/authService";

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  const handleLogout = async () => {
    try {
      await logoutUser();
    } catch (error) {
      console.error("Logout error:", error);
    }

    localStorage.removeItem("user");
    navigate("/login");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#1f1f1f",
        color: "white",
        padding: "40px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <h1 style={{ fontSize: "42px", marginBottom: "20px" }}>
        Welcome, {user.name || "User"}
      </h1>

      {user.photo && (
        <img
          src={user.photo}
          alt="profile"
          style={{
            width: "90px",
            height: "90px",
            borderRadius: "50%",
            marginBottom: "20px",
          }}
        />
      )}

      <p style={{ marginBottom: "30px", fontSize: "18px" }}>
        {user.email || "No email"}
      </p>

      <h2 style={{ marginBottom: "20px" }}>Dashboard</h2>

      <div>
        <button onClick={() => navigate("/student")} style={btnStyle}>
          Face Registration
        </button>

        <button onClick={() => navigate("/attendance")} style={btnStyle}>
          Attendance
        </button>

        <button onClick={() => navigate("/admin")} style={btnStyle}>
          Admin Panel
        </button>
      </div>

      <button onClick={handleLogout} style={logoutStyle}>
        Logout
      </button>
    </div>
  );
};

const btnStyle: React.CSSProperties = {
  marginRight: "12px",
  marginBottom: "12px",
  padding: "12px 22px",
  borderRadius: "8px",
  border: "none",
  backgroundColor: "#cddc39",
  color: "#1f1f1f",
  fontWeight: "bold",
  cursor: "pointer",
};

const logoutStyle: React.CSSProperties = {
  marginTop: "40px",
  padding: "12px 26px",
  borderRadius: "8px",
  border: "none",
  backgroundColor: "#ffffff",
  color: "#1f1f1f",
  fontWeight: "bold",
  cursor: "pointer",
};

export default HomePage;