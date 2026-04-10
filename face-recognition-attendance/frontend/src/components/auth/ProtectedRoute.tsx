import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";

type ProtectedRouteProps = {
  children: ReactNode;
  requiredRole?: "admin" | "student";
};

const ProtectedRoute = ({ children, requiredRole }: ProtectedRouteProps) => {
  const rawUser = localStorage.getItem("user");

  if (!rawUser) {
    return <Navigate to="/login" replace />;
  }

  const user = JSON.parse(rawUser);

  if (requiredRole && user.role !== requiredRole) {
    return <Navigate to="/" replace />;
  }

  return children;
};

export default ProtectedRoute;
