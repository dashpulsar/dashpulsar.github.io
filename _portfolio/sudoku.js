
let activeCell = null;

function generateGrid() {
    const container = document.querySelector('.grid');
    container.innerHTML = '';
    for (let i = 0; i < 81; i++) {
        const button = document.createElement('button');
        button.onclick = function() {
            if (!this.classList.contains('fixed')) {
                if (activeCell) {
                    activeCell.classList.remove('active');
                }
                activeCell = this;
                this.classList.add('active');
            }
        };
        container.appendChild(button);
    }
}

function generateSudoku() {
    const size = 9;
    const board = new Array(size).fill().map(() => new Array(size).fill(''));
    generateGrid();

    function isValid(board, row, col, num) {
        for (let i = 0; i < size; i++) {
            const boxRow = 3 * Math.floor(row / 3) + Math.floor(i / 3);
            const boxCol = 3 * Math.floor(col / 3) + i % 3;
            if (board[row][i] == num || board[i][col] == num || board[boxRow][boxCol] == num) {
                return false;
            }
        }
        return true;
    }

    function solve(board, row, col) {
        if (col == size) {
            col = 0;
            row++;
        }
        if (row == size) {
            return true;
        }
        if (board[row][col] != '') {
            return solve(board, row, col + 1);
        }
        for (let num = 1; num <= size; num++) {
            if (isValid(board, row, col, num)) {
                board[row][col] = num;
                if (solve(board, row, col + 1)) {
                    return true;
                }
                board[row][col] = '';
            }
        }
        return false;
    }

    solve(board, 0, 0);

    let cells = document.querySelectorAll('.grid button');
    cells.forEach((cell, index) => {
        const row = Math.floor(index / 9);
        const col = index % 9;
        cell.textContent = board[row][col];
        let shouldRemove = Math.random() > 0.8;
        if (shouldRemove) {
            cell.textContent = '';
        } else {
            cell.classList.add('fixed', 'filled');
        }
    });
}

function setupNumberSelector() {
    const selector = document.querySelector('.number-selector');
    selector.innerHTML = '';
    for (let i = 1; i <= 9; i++) {
        const numButton = document.createElement('button');
        numButton.textContent = i;
        numButton.onclick = function() {
            if (activeCell && !activeCell.classList.contains('fixed')) {
                activeCell.textContent = i;
                activeCell.classList.add('filled');
                activeCell.classList.remove('active');
                activeCell = null;
            }
        };
        selector.appendChild(numButton);
    }
    const clearButton = document.createElement('button');
    clearButton.textContent = 'X';
    clearButton.onclick = function() {
        if (activeCell && !activeCell.classList.contains('fixed')) {
            activeCell.textContent = '';
            activeCell.classList.remove('filled', 'active');
            activeCell = null;
        }
    };
    selector.appendChild(clearButton);

    const newgame = document.createElement('button');
    newgame.textContent="New";
    newgame.onclick =generateSudoku;
    selector.appendChild(newgame);

    const checkSudoku = document.createElement('button');
    checkSudoku.textContent = "Go";
    checkSudoku.onclick = function(){
        const cells = document.querySelectorAll('.grid button');
        let isComplete = true;
        let board = [];

        cells.forEach((cell, index) => {
            if (!cell.textContent.trim()) {
                isComplete = false;
            }
            const row = Math.floor(index / 9);
            const col = index % 9;
            if (!board[row]) {
                board[row] = [];
            }
            board[row][col] = cell.textContent || '.';
        });

        if (!isComplete) {
            alert("Not Finished");
            return;
        }

        if (isValidSudoku(board)) {
            alert("Success! Fxxk you kom1sch!");
        } else {
            alert("There are errors in your solution.");
        }
    }
    selector.appendChild(checkSudoku);

    function isValidSudoku(board) {
        const size = 9;
        const rows = new Array(size).fill().map(() => new Set());
        const cols = new Array(size).fill().map(() => new Set());
        const boxes = new Array(size).fill().map(() => new Set());

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const num = board[i][j];
                if (num === '.') continue;

                const boxIndex = Math.floor(i / 3) * 3 + Math.floor(j / 3);
                if (rows[i].has(num) || cols[j].has(num) || boxes[boxIndex].has(num)) {
                    return false;
                }
                rows[i].add(num);
                cols[j].add(num);
                boxes[boxIndex].add(num);
            }
        }
        return true;

    }
}

document.addEventListener('DOMContentLoaded', function() {
    generateSudoku();
    setupNumberSelector();
});