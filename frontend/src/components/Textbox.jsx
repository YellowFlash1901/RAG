import { TextField, IconButton, Tooltip } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';

const Textbox = () => {
  return (
    <div style={{ 
      position: 'fixed',
      bottom: 0,
      left: 0,
      right: 0,
      padding: '1rem',
      backgroundColor: 'white',
      zIndex: 1000,
      display: 'flex',
      justifyContent: 'center',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        width: '80%',
        maxWidth: '1200px',
      }}>
        <Tooltip title="Upload file">
          <IconButton color="primary" component="label">
            <input hidden accept="*/*" type="file" />
            <AttachFileIcon />
          </IconButton>
        </Tooltip>

        <TextField 
          id="outlined-basic" 
          placeholder="Ask a question..."
          variant="outlined"
          fullWidth 
          size="medium"
        />

        <Tooltip title="Send message">
          <IconButton color="primary">
            <SendIcon />
          </IconButton>
        </Tooltip>
      </div>
    </div>
  );
};

export default Textbox;
