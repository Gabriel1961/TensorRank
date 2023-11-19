import { Navigate } from "@solidjs/router";
import { ParentComponent } from "solid-js";
import Spinner from "../components/Spinner";
import { useLogin } from "./loginContext";

const ProtectedRoute: ParentComponent = (props) => {
  const [getLoginContext] = useLogin()
  if(getLoginContext().loading)
    return <Spinner/>
  return getLoginContext().user ? props.children : <Navigate href="./"/>
} 

export default ProtectedRoute;